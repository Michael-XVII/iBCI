"""SPINT decoder wrapper with pluggable streaming calibration encoder."""
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.spint import (
    CachedDecoupledMultiLayerCrossAttention,
    DecoupledKVState,
    SpintModel,
)
from src.models.components.streaming_encoders import CalibrationEncoder


class CalibrationFixedSlotRouter(nn.Module):
    """Map a variable-size NeuronID-conditioned set to fixed decoder tokens.

    Calibration derives session-specific routing and FiLM parameters from the
    per-unit identities.  The same state can then be reused to project every
    live neural window from that session into a fixed number of slot tokens.
    """

    def __init__(
        self,
        window_size: int,
        slot_count: int,
        router_dim: int,
        routing_mode: str,
        fusion: str,
        temperature: float,
    ) -> None:
        super().__init__()
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if router_dim <= 0:
            raise ValueError("router_dim must be positive")
        if routing_mode not in {"soft", "top1"}:
            raise ValueError("routing_mode must be 'soft' or 'top1'")
        if fusion not in {"additive", "film"}:
            raise ValueError("fusion must be 'additive' or 'film'")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.window_size = window_size
        self.slot_count = slot_count
        self.router_dim = router_dim
        self.routing_mode = routing_mode
        self.fusion = fusion
        self.temperature = temperature
        self.key_projection = nn.Linear(window_size, router_dim, bias=False)
        self.slot_queries = nn.Parameter(torch.empty(slot_count, router_dim))
        self.mass_projection = nn.Linear(1, window_size)
        self.gain_projection = nn.Linear(window_size, window_size)
        self.bias_projection = nn.Linear(window_size, window_size)
        nn.init.normal_(self.slot_queries, std=router_dim**-0.5)
        nn.init.zeros_(self.mass_projection.weight)
        nn.init.zeros_(self.mass_projection.bias)
        nn.init.zeros_(self.gain_projection.weight)
        nn.init.zeros_(self.gain_projection.bias)
        nn.init.zeros_(self.bias_projection.weight)
        nn.init.zeros_(self.bias_projection.bias)

    def _expand_identity(self, identity: torch.Tensor, num_neurons: int) -> torch.Tensor:
        if identity.ndim != 3:
            raise ValueError(f"Expected identity [B,N,W], got {tuple(identity.shape)}")
        if identity.shape[-1] != self.window_size:
            raise ValueError(
                f"Identity window {identity.shape[-1]} does not match router window {self.window_size}"
            )
        if identity.shape[1] == num_neurons:
            return identity
        if identity.shape[1] == 1:
            return identity.expand(-1, num_neurons, -1)
        raise ValueError(
            f"Identity neuron dimension {identity.shape[1]} does not match neural dimension {num_neurons}"
        )

    def derive_calibration_state(self, identity: torch.Tensor, num_neurons: int) -> dict[str, torch.Tensor]:
        """Derive a serializable session routing state from NeuronID outputs."""
        identity = self._expand_identity(identity, num_neurons)
        keys = self.key_projection(identity)
        logits = torch.einsum("bnd,kd->bnk", keys, self.slot_queries)
        logits = logits / (self.router_dim**0.5 * self.temperature)
        soft_assignment = torch.softmax(logits, dim=-1)
        if self.routing_mode == "top1":
            indices = soft_assignment.argmax(dim=-1)
            hard_assignment = torch.nn.functional.one_hot(
                indices, num_classes=self.slot_count
            ).to(dtype=soft_assignment.dtype)
            assignment = (
                hard_assignment + soft_assignment - soft_assignment.detach()
                if self.training
                else hard_assignment
            )
        else:
            assignment = soft_assignment

        slot_mass = assignment.sum(dim=1)
        normalized_assignment = assignment.transpose(1, 2) / slot_mass.unsqueeze(-1).clamp_min(1.0e-6)
        slot_identity = torch.einsum("bkn,bnw->bkw", normalized_assignment, identity)
        mass_feature = torch.log1p(slot_mass).unsqueeze(-1)
        conditioning = slot_identity + self.mass_projection(mass_feature)
        if self.fusion == "additive":
            gain = torch.ones_like(conditioning)
            bias = conditioning
        else:
            gain = 1.0 + 0.5 * torch.tanh(self.gain_projection(conditioning))
            bias = self.bias_projection(conditioning)
        return {
            "assignment": assignment,
            "normalized_assignment": normalized_assignment,
            "slot_mass": slot_mass,
            "slot_identity": slot_identity,
            "gain": gain,
            "bias": bias,
        }

    def project_neural(
        self,
        neural_tokens: torch.Tensor,
        calibration_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project ``[B,N,W]`` live spikes into fixed ``[B,K,W]`` slot tokens."""
        if neural_tokens.ndim != 3:
            raise ValueError(f"Expected neural tokens [B,N,W], got {tuple(neural_tokens.shape)}")
        if neural_tokens.shape[-1] != self.window_size:
            raise ValueError(
                f"Neural window {neural_tokens.shape[-1]} does not match router window {self.window_size}"
            )
        normalized_assignment = calibration_state["normalized_assignment"]
        state_batch_size = normalized_assignment.shape[0]
        neural_batch_size = neural_tokens.shape[0]
        if state_batch_size == 1 and neural_batch_size > 1:
            normalized_assignment = normalized_assignment.expand(neural_batch_size, -1, -1)
            gain = calibration_state["gain"].expand(neural_batch_size, -1, -1)
            bias = calibration_state["bias"].expand(neural_batch_size, -1, -1)
            neuron_gate = calibration_state.get("neuron_gate")
            if neuron_gate is not None:
                neuron_gate = neuron_gate.expand(neural_batch_size, -1, -1)
        elif state_batch_size == neural_batch_size:
            gain = calibration_state["gain"]
            bias = calibration_state["bias"]
            neuron_gate = calibration_state.get("neuron_gate")
        else:
            raise ValueError(
                "Calibration-state batch size must match neural batch size, unless a single "
                "session state is reused for multiple online windows"
            )
        if normalized_assignment.shape[-1] != neural_tokens.shape[1]:
            raise ValueError("Calibration-state unit count must match neural unit count")
        if neuron_gate is not None:
            expected_gate_shape = (*neural_tokens.shape[:2], 1)
            if neuron_gate.shape != expected_gate_shape:
                raise ValueError(
                    "Calibration-state neuron gate must have shape "
                    f"{expected_gate_shape}, got {tuple(neuron_gate.shape)}"
                )
            neural_tokens = neural_tokens * neuron_gate
        slot_neural = torch.einsum("bkn,bnw->bkw", normalized_assignment, neural_tokens)
        return gain * slot_neural + bias

    def forward(self, neural_tokens: torch.Tensor, identity: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        calibration_state = self.derive_calibration_state(identity, neural_tokens.shape[1])
        return self.project_neural(neural_tokens, calibration_state), calibration_state


class StreamingSpintModel(nn.Module):
    """Frozen decoder + trainable/streaming calibration encoder."""

    def __init__(
        self,
        decoder: SpintModel,
        id_encoder: CalibrationEncoder,
        fixed_slot_count: int = 0,
        fixed_slot_dim: int = 32,
        fixed_slot_mode: str = "soft",
        fixed_slot_fusion: str = "film",
        fixed_slot_temperature: float = 1.0,
        decoder_mode: Literal["coupled", "decoupled"] = "coupled",
        decoupled_key_mode: Literal[
            "e_t4", "e_ts4", "e_only", "x_only"
        ] = "e_t4",
        decoupled_key_dim: int = 32,
        decoupled_value_dim: int = 32,
        decoupled_num_heads: int = 2,
        decoupled_direct_feature_dim: int = 4,
        reliability_logit_bias: bool = False,
        reliability_gamma_init: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if decoder_mode not in {"coupled", "decoupled"}:
            raise ValueError("decoder_mode must be 'coupled' or 'decoupled'")
        if decoupled_key_mode not in {"e_t4", "e_ts4", "e_only", "x_only"}:
            raise ValueError(
                "decoupled_key_mode must be one of "
                "{'e_t4','e_ts4','e_only','x_only'}"
            )
        if decoupled_direct_feature_dim <= 0:
            raise ValueError("decoupled_direct_feature_dim must be positive")
        if reliability_gamma_init <= 0.0:
            raise ValueError("reliability_gamma_init must be positive")
        if reliability_logit_bias and decoder_mode != "coupled":
            raise ValueError("reliability logit bias requires coupled decoder attention")
        if decoder_mode == "decoupled" and fixed_slot_count > 0:
            raise ValueError(
                "decoupled K/V requires fixed_slot_count=0 so unit keys remain explicit"
            )
        if decoder_mode == "decoupled" and decoder.num_layers != 1:
            raise ValueError("The state-neutral decoupled K/V pilot requires num_layers=1")
        self.decoder = decoder
        self.id_encoder = id_encoder
        self.window_size = decoder.window_size
        self._decoder_frozen = False
        self.decoder_mode = decoder_mode
        self.decoupled_key_mode = decoupled_key_mode
        self.decoupled_direct_feature_dim = decoupled_direct_feature_dim
        self.reliability_logit_bias = bool(reliability_logit_bias)
        self.reliability_gamma_init = float(reliability_gamma_init)
        # softplus makes gamma non-negative; its inverse makes initialization
        # near-neutral without a target-side hyperparameter search.
        raw_init = float(torch.log(torch.expm1(torch.tensor(reliability_gamma_init))).item())
        gamma_raw = torch.full((decoder.num_layers,), raw_init, dtype=torch.float32)
        if self.reliability_logit_bias:
            self.reliability_logit_gamma_raw = nn.Parameter(gamma_raw)
        else:
            # Preserve non-E04 trainable/total parameter counts and behavior.
            self.register_buffer("reliability_logit_gamma_raw", gamma_raw)
        self.fixed_slot_router = (
            CalibrationFixedSlotRouter(
                window_size=decoder.window_size,
                slot_count=fixed_slot_count,
                router_dim=fixed_slot_dim,
                routing_mode=fixed_slot_mode,
                fusion=fixed_slot_fusion,
                temperature=fixed_slot_temperature,
            )
            if fixed_slot_count > 0
            else None
        )
        self.decoupled_transformer = (
            CachedDecoupledMultiLayerCrossAttention(
                num_layers=1,
                d_model=decoder.model_dim,
                nhead=decoupled_num_heads,
                key_input_dim=decoder.window_size + decoupled_direct_feature_dim,
                value_input_dim=decoder.window_size,
                key_dim=decoupled_key_dim,
                value_dim=decoupled_value_dim,
                dim_feedforward=decoder.transformer.layers[0].ffn[0].out_features,
                dropout=decoder.tf_drop_rate,
                cache_reference_width=decoder.window_size,
            )
            if decoder_mode == "decoupled"
            else None
        )
        if self.decoupled_transformer is not None:
            # Transfer every shape-compatible transformer-base tensor from the
            # immutable teacher decoder. Q/K/V/out projections and input norms
            # remain the only newly initialized decoupled parameters.
            legacy = decoder.transformer.layers[0]
            decoupled = self.decoupled_transformer.layers[0]
            decoupled.norm1.load_state_dict(legacy.norm1.state_dict(), strict=True)
            decoupled.norm2.load_state_dict(legacy.norm2.state_dict(), strict=True)
            decoupled.ffn.load_state_dict(legacy.ffn.state_dict(), strict=True)

    def compute_identity(
        self,
        calib_trials: torch.Tensor,
        side_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.id_encoder.forward_batch(
            calib_trials, side_features=side_features, electrode_ids=electrode_ids
        )

    def reliability_logit_gammas(self) -> torch.Tensor:
        if not self.reliability_logit_bias:
            return self.reliability_logit_gamma_raw.new_zeros((self.decoder.num_layers,))
        return F.softplus(self.reliability_logit_gamma_raw)

    def decode_with_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        neuron_gate: torch.Tensor | None = None,
        reliability_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """neural: [B,W,N], identity: [B,N,W] -> behavior [B,W,C].

        Frozen decoder weights stay in eval() with requires_grad=False, but the
        forward path must remain differentiable w.r.t. identity E.
        """
        src = neural.permute(0, 2, 1)
        if neuron_gate is not None:
            src = src * neuron_gate
        if self.fixed_slot_router is None:
            src = src + identity
        else:
            src, _ = self.fixed_slot_router(src, identity)

        if self._decoder_frozen:
            self.decoder.eval()

        batch_size = src.size(0)
        num_neurons = src.size(1)
        dropout_mask = torch.ones(batch_size, num_neurons, device=src.device, dtype=src.dtype)
        if not self._decoder_frozen:
            if self.decoder.dynamic_dropout and self.training:
                import random

                p = random.uniform(self.decoder.dynamic_dropout_low, self.decoder.dynamic_dropout_high)
                dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=True)
            elif self.decoder.dropout_rate > 0.0 and self.training:
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=self.decoder.dropout_rate, training=True
                )
        src = src * dropout_mask.unsqueeze(-1)

        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        if self.reliability_logit_bias:
            if reliability_q is None or reliability_q.shape != src.shape[:2]:
                raise ValueError("E04 reliability_q must have shape [B,N]")
            if not torch.isfinite(reliability_q).all():
                raise ValueError("E04 reliability_q must be finite")
        elif reliability_q is not None:
            raise ValueError("reliability_q requires reliability_logit_bias=True")
        transformer_output, _ = self.decoder.transformer(
            rep.repeat(src.size(0), 1, 1), src,
            attention_logit_bias=reliability_q,
            logit_bias_gammas=(self.reliability_logit_gammas() if self.reliability_logit_bias else None),
        )
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def _expanded_identity(
        self, identity: torch.Tensor, batch_size: int, num_neurons: int
    ) -> torch.Tensor:
        if identity.ndim != 3 or identity.shape[-1] != self.window_size:
            raise ValueError(
                f"identity must have shape [B,N,{self.window_size}], "
                f"got {tuple(identity.shape)}"
            )
        if identity.shape[0] == 1 and batch_size > 1:
            identity = identity.expand(batch_size, -1, -1)
        if identity.shape[:2] != (batch_size, num_neurons):
            raise ValueError(
                "identity batch/unit dimensions must match neural input; got "
                f"{tuple(identity.shape[:2])} vs {(batch_size, num_neurons)}"
            )
        return identity

    def _decoupled_key_input(
        self,
        identity: torch.Tensor,
        decoder_key_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if decoder_key_features is None:
            if self.decoupled_key_mode != "e_only":
                raise ValueError(
                    f"{self.decoupled_key_mode} requires decoder_key_features"
                )
            decoder_key_features = identity.new_zeros(
                *identity.shape[:2], self.decoupled_direct_feature_dim
            )
        if decoder_key_features.ndim != 3:
            raise ValueError(
                "decoder_key_features must have shape [B,N,D], got "
                f"{tuple(decoder_key_features.shape)}"
            )
        if decoder_key_features.shape[:2] != identity.shape[:2]:
            raise ValueError(
                "decoder_key_features batch/unit dimensions must match identity"
            )
        if decoder_key_features.shape[-1] != self.decoupled_direct_feature_dim:
            raise ValueError(
                "decoder_key_features last dimension must be "
                f"{self.decoupled_direct_feature_dim}, got "
                f"{decoder_key_features.shape[-1]}"
            )
        return torch.cat([identity, decoder_key_features], dim=-1)

    def derive_decoupled_kv_state(
        self,
        identity: torch.Tensor,
        decoder_key_features: torch.Tensor | None = None,
    ) -> DecoupledKVState:
        """Build the calibration-only projected K cache.

        This method intentionally remains differentiable for training. A
        deployment caller may invoke it under ``torch.no_grad()`` and discard
        ``identity``/``decoder_key_features`` immediately afterward.
        """
        if self.decoupled_transformer is None:
            raise RuntimeError("derive_decoupled_kv_state requires decoder_mode='decoupled'")
        if self.decoupled_key_mode == "x_only":
            raise RuntimeError("x_only uses a dynamic activity key and has no static K state")
        key_input = self._decoupled_key_input(identity, decoder_key_features)
        return self.decoupled_transformer.derive_static_key(key_input)

    def _apply_decoder_neuron_dropout(self, src: torch.Tensor) -> torch.Tensor:
        batch_size, num_neurons = src.shape[:2]
        dropout_mask = torch.ones(
            batch_size, num_neurons, device=src.device, dtype=src.dtype
        )
        if not self._decoder_frozen:
            if self.decoder.dynamic_dropout and self.training:
                import random

                probability = random.uniform(
                    self.decoder.dynamic_dropout_low,
                    self.decoder.dynamic_dropout_high,
                )
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=probability, training=True
                )
            elif self.decoder.dropout_rate > 0.0 and self.training:
                dropout_mask = torch.nn.functional.dropout(
                    dropout_mask, p=self.decoder.dropout_rate, training=True
                )
        return src * dropout_mask.unsqueeze(-1)

    def decode_with_decoupled_kv_state(
        self,
        neural: torch.Tensor,
        state: DecoupledKVState,
    ) -> torch.Tensor:
        """Online V(x) decode using a previously cached static K."""
        if self.decoupled_transformer is None:
            raise RuntimeError(
                "decode_with_decoupled_kv_state requires decoder_mode='decoupled'"
            )
        if self.decoupled_key_mode == "x_only":
            raise RuntimeError("x_only has no static K state")
        src = self._apply_decoder_neuron_dropout(neural.permute(0, 2, 1))
        if self._decoder_frozen:
            self.decoder.eval()
            self.decoupled_transformer.eval()
        query = self.decoder.fc_in(self.decoder.rep).to(src)
        query = query.repeat(src.shape[0], 1, 1)
        transformer_output, _ = self.decoupled_transformer.forward_cached(
            query, state, src
        )
        return self.decoder.fc_out(transformer_output).permute(0, 2, 1)

    def decode_with_decoupled_identity(
        self,
        neural: torch.Tensor,
        identity: torch.Tensor,
        decoder_key_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training/reference path for static-identity or dynamic-activity keys."""
        if self.decoupled_transformer is None:
            raise RuntimeError(
                "decode_with_decoupled_identity requires decoder_mode='decoupled'"
            )
        src = self._apply_decoder_neuron_dropout(neural.permute(0, 2, 1))
        batch_size, num_neurons = src.shape[:2]
        identity = self._expanded_identity(identity, batch_size, num_neurons)
        if self._decoder_frozen:
            self.decoder.eval()
            self.decoupled_transformer.eval()
        query = self.decoder.fc_in(self.decoder.rep).to(src)
        query = query.repeat(batch_size, 1, 1)
        if self.decoupled_key_mode == "x_only":
            zeros = src.new_zeros(
                batch_size, num_neurons, self.decoupled_direct_feature_dim
            )
            key_input = torch.cat([src, zeros], dim=-1)
            transformer_output, _ = self.decoupled_transformer(
                query, key_input, src
            )
        else:
            state = self.derive_decoupled_kv_state(
                identity, decoder_key_features=decoder_key_features
            )
            transformer_output, _ = self.decoupled_transformer.forward_cached(
                query, state, src
            )
        return self.decoder.fc_out(transformer_output).permute(0, 2, 1)

    def decoupled_cost_receipt(
        self, *, batch_size: int, num_neurons: int
    ) -> dict[str, object]:
        if self.decoupled_transformer is None:
            raise RuntimeError("decoupled_cost_receipt requires decoder_mode='decoupled'")
        receipt = self.decoupled_transformer.online_cost_receipt(
            batch_size=batch_size,
            num_queries=self.decoder.num_covariates,
            num_units=num_neurons,
        )
        receipt["key_mode"] = self.decoupled_key_mode
        receipt["static_key_cache_applicable"] = self.decoupled_key_mode != "x_only"
        receipt["fixed_slot_count"] = 0
        return receipt

    def decoder_cost_comparison_receipt(
        self, *, batch_size: int = 1, num_neurons: int = 64
    ) -> dict[str, object]:
        """Exact configured-MAC comparison for legacy coupled vs decoupled paths.

        Counts the Linear/attention/FFN MACs executed by this source, including
        the query read-in and output projection shared by both paths. Elementwise
        activations, norms, softmax and additions are reported as exclusions.
        """
        if batch_size <= 0 or num_neurons <= 0:
            raise ValueError("batch_size and num_neurons must be positive")
        decoder = self.decoder
        batch = batch_size
        units = num_neurons
        covariates = decoder.num_covariates
        window = decoder.window_size
        model_dim = decoder.model_dim
        feedforward = decoder.transformer.layers[0].ffn[0].out_features

        query_readin = batch * covariates * (
            window * model_dim + model_dim * model_dim
        )
        ffn = batch * covariates * 2 * model_dim * feedforward
        output_readout = batch * covariates * model_dim * window
        source_readin = batch * units * (
            window * model_dim + model_dim * model_dim
        )
        coupled_qkv = batch * (
            covariates * model_dim * model_dim
            + 2 * units * model_dim * model_dim
        )
        coupled_attention_output = (
            batch * covariates * model_dim * model_dim
        )
        coupled_scores = batch * covariates * units * model_dim
        coupled_weighted_values = coupled_scores
        coupled_total = (
            source_readin
            + query_readin
            + coupled_qkv
            + coupled_attention_output
            + coupled_scores
            + coupled_weighted_values
            + ffn
            + output_readout
        )
        coupled = {
            "source_readin": source_readin,
            "query_readin": query_readin,
            "qkv_projections": coupled_qkv,
            "attention_output_projection": coupled_attention_output,
            "qk_scores": coupled_scores,
            "weighted_values": coupled_weighted_values,
            "ffn": ffn,
            "output_readout": output_readout,
            "total": coupled_total,
            "persistent_state_width": window,
            "persistent_state_bytes_fp32": batch * units * window * 4,
        }
        result: dict[str, object] = {
            "schema_version": 1,
            "reference_shape": {
                "batch_size": batch,
                "num_units": units,
                "num_covariates": covariates,
                "window_size": window,
                "model_dim": model_dim,
                "feedforward_dim": feedforward,
            },
            "coupled": coupled,
            "counted_operations": "Linear, attention matmul, and FFN MACs",
            "excluded_operations": [
                "elementwise_add",
                "activation",
                "normalization",
                "softmax",
                "dropout",
            ],
        }
        if self.decoupled_transformer is None:
            result["active_mode"] = "coupled"
            return result

        core = self.decoupled_transformer.online_cost_receipt(
            batch_size=batch,
            num_queries=covariates,
            num_units=units,
        )
        core_online = core["online_macs_per_frame"]
        assert isinstance(core_online, dict)
        decoupled_total = (
            query_readin
            + int(core_online["query_projection"])
            + int(core_online["value_projection"])
            + int(core_online["qk_scores"])
            + int(core_online["weighted_values"])
            + int(core_online["output_projection"])
            + ffn
            + output_readout
        )
        static_key = self.decoupled_key_mode != "x_only"
        state_width = (
            self.decoupled_transformer.num_layers
            * self.decoupled_transformer.key_dim
            if static_key
            else 0
        )
        result["active_mode"] = "decoupled"
        result["decoupled"] = {
            "key_mode": self.decoupled_key_mode,
            "query_readin": query_readin,
            **core_online,
            "ffn": ffn,
            "output_readout": output_readout,
            "total": decoupled_total,
            "online_mac_reduction_fraction_vs_coupled": (
                1.0 - decoupled_total / coupled_total
            ),
            "persistent_state_width": state_width,
            "persistent_state_bytes_fp32": batch * units * state_width * 4,
            "persistent_state_nonincreasing_vs_coupled": state_width <= window,
            "static_key_projection_calibration_only_macs": core[
                "calibration_only_macs"
            ]["static_key_projection"],
            "no_unit_quadratic_term": core_online["no_unit_quadratic_term"],
        }
        return result

    @torch.no_grad()
    def derive_fixed_slot_state(
        self,
        identity: torch.Tensor,
        num_neurons: int,
        neuron_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Export the calibration-only state required by a fixed-slot deployment."""
        if self.fixed_slot_router is None:
            raise RuntimeError("derive_fixed_slot_state requires fixed_slot_count > 0")
        calibration_state = self.fixed_slot_router.derive_calibration_state(identity, num_neurons)
        if neuron_gate is not None:
            expected_gate_shape = (identity.shape[0], num_neurons, 1)
            if neuron_gate.shape != expected_gate_shape:
                raise ValueError(
                    f"Neuron gate must have shape {expected_gate_shape}, got {tuple(neuron_gate.shape)}"
                )
            calibration_state["neuron_gate"] = neuron_gate
        return calibration_state

    @torch.no_grad()
    def decode_with_fixed_slot_state(
        self,
        neural: torch.Tensor,
        calibration_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Decode with a previously derived fixed-slot state without recomputing identities."""
        if self.fixed_slot_router is None:
            raise RuntimeError("decode_with_fixed_slot_state requires fixed_slot_count > 0")
        src = neural.permute(0, 2, 1)
        src = self.fixed_slot_router.project_neural(src, calibration_state)
        if self._decoder_frozen:
            self.decoder.eval()
        src = self.decoder.fc_in(src)
        rep = self.decoder.fc_in(self.decoder.rep).to(src)
        transformer_output, _ = self.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        output = self.decoder.fc_out(transformer_output)
        return output.permute(0, 2, 1)

    def forward(
        self,
        neural: torch.Tensor,
        calib_trials: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        side_features: Optional[torch.Tensor] = None,
        decoder_key_features: Optional[torch.Tensor] = None,
        electrode_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neuron_gate = None
        reliability_q = None
        encoder_side_features = side_features
        if self.reliability_logit_bias:
            if side_features is None or side_features.ndim != 3 or side_features.shape[-1] != 5:
                raise ValueError("E04 requires normalized side_features [B,N,5]=[T4R,q_theta]")
            encoder_side_features = side_features[..., :4]
            reliability_q = side_features[..., 4]
        if identity is None:
            if calib_trials is None:
                raise ValueError("Either calib_trials or identity must be provided")
            if hasattr(self.id_encoder, "forward_batch_with_gate"):
                identity, neuron_gate = self.id_encoder.forward_batch_with_gate(
                    calib_trials, side_features=encoder_side_features
                )
            else:
                identity = self.compute_identity(
                    calib_trials,
                    side_features=encoder_side_features,
                    electrode_ids=electrode_ids,
                )
        if self.decoder_mode == "coupled":
            behavior = self.decode_with_identity(
                neural, identity, neuron_gate=neuron_gate, reliability_q=reliability_q
            )
        else:
            if neuron_gate is not None:
                raise ValueError(
                    "decoupled K/V does not support encoder-provided neuron gates"
                )
            behavior = self.decode_with_decoupled_identity(
                neural,
                identity,
                decoder_key_features=decoder_key_features,
            )
        return behavior, identity

    def freeze_decoder(self) -> int:
        frozen = 0
        for param in self.decoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
        if self.decoupled_transformer is not None:
            for param in self.decoupled_transformer.parameters():
                param.requires_grad = False
                frozen += param.numel()
        self._decoder_frozen = True
        self.decoder.eval()
        if self.decoupled_transformer is not None:
            self.decoupled_transformer.eval()
        return frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._decoder_frozen:
            self.decoder.eval()
            if self.decoupled_transformer is not None:
                self.decoupled_transformer.eval()
        return self

    def trainable_encoder_parameters(self):
        return (parameter for parameter in self.id_encoder.parameters() if parameter.requires_grad)
