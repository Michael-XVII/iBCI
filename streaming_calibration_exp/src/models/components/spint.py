"""SPINT model architecture - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Scaffolding adapted from the Hydra template (ashleve/lightning-hydra-template).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        # Pre-norm before cross-attention
        query_norm = self.norm1(query)
        key_value_norm = self.norm1(key_value)
        cross_attn_output, attn_score = self.cross_attn(
            query=query_norm,
            key=key_value_norm,
            value=key_value_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = query + self.dropout(cross_attn_output)

        # Pre-norm before feedforward
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x, attn_score


class MultiLayerCrossAttention(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self, query, key_value, attn_mask=None, key_padding_mask=None,
        attention_logit_bias=None, logit_bias_gammas=None,
    ):
        """Optionally add a per-unit bias immediately before attention softmax.

        ``attention_logit_bias`` is [B,N] and expands over covariate queries and
        heads. It never enters key/value inputs, so reliability cannot alter the
        activity or identity representations.
        """
        if (attention_logit_bias is None) != (logit_bias_gammas is None):
            raise ValueError("attention_logit_bias and logit_bias_gammas must be provided together")
        if attention_logit_bias is not None:
            if (attention_logit_bias.ndim != 2 or attention_logit_bias.shape[0] != query.shape[0]
                    or attention_logit_bias.shape[1] != key_value.shape[1]):
                raise ValueError("attention_logit_bias must have shape [B,N] matching query/key batch")
            if len(logit_bias_gammas) != len(self.layers):
                raise ValueError("logit_bias_gammas must provide one scalar per attention layer")
        x = query
        attn_scores = []
        for layer_index, layer in enumerate(self.layers):
            layer_mask = attn_mask
            if attention_logit_bias is not None:
                if attn_mask is not None:
                    raise ValueError("E04 reliability bias does not combine with an arbitrary attention mask")
                bias = logit_bias_gammas[layer_index] * attention_logit_bias
                layer_mask = bias[:, None, None, :].expand(
                    -1, layer.cross_attn.num_heads, x.shape[1], -1
                ).reshape(-1, x.shape[1], key_value.shape[1])
            x, attn_score = layer(x, key_value, attn_mask=layer_mask, key_padding_mask=key_padding_mask)
            attn_scores.append(attn_score)
        return x, attn_scores


@dataclass(frozen=True)
class DecoupledKVState:
    """Persistent session state for cached decoupled cross-attention.

    The state deliberately owns *only* projected static keys.  In particular it
    does not retain an identity embedding, side features, activity values, raw
    calibration samples, or attention scores.  That boundary is important for
    the streaming memory receipt: after calibration, the caller may discard the
    inputs used to construct this object.
    """

    projected_keys: Tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.projected_keys:
            raise ValueError("DecoupledKVState requires at least one projected key tensor")
        for projected_key in self.projected_keys:
            if projected_key.ndim != 3:
                raise ValueError(
                    "Each projected static key must have shape [B, N, Dk], "
                    f"got {tuple(projected_key.shape)}"
                )

    @property
    def projected_key(self) -> torch.Tensor:
        """The single cached key tensor for the guarded one-layer pilot."""
        if len(self.projected_keys) != 1:
            raise RuntimeError("projected_key is only unambiguous for a one-layer state")
        return self.projected_keys[0]

    @property
    def nbytes(self) -> int:
        """Total bytes occupied by the persistent tensors in this state."""
        return sum(key.numel() * key.element_size() for key in self.projected_keys)


class CachedDecoupledCrossAttentionLayer(nn.Module):
    """Cross-attention with separate static keys and online activity values.

    This is intentionally separate from :class:`CrossAttentionLayer`, rather
    than changing that class's ``key_value`` API.  Existing SPINT checkpoints
    and the legacy coupled path consequently retain their exact module layout
    and behavior.

    ``derive_static_key`` is calibration-time work.  ``forward_cached`` never
    calls ``key_proj`` and consumes only the projected key cache plus online
    values.  The public attention-score contract is ``[B, C, N]`` (heads are
    averaged only for reporting; the actual weighted values remain multi-head).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 2,
        key_input_dim: int = 54,
        value_input_dim: int = 50,
        key_dim: int = 32,
        value_dim: int = 32,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or nhead <= 0:
            raise ValueError("d_model and nhead must both be positive")
        if key_input_dim <= 0 or value_input_dim <= 0:
            raise ValueError("key_input_dim and value_input_dim must both be positive")
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError("key_dim and value_dim must both be positive")
        if key_dim % nhead != 0 or value_dim % nhead != 0:
            raise ValueError("key_dim and value_dim must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.key_input_dim = key_input_dim
        self.value_input_dim = value_input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_head_dim = key_dim // nhead
        self.value_head_dim = value_dim // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(key_input_dim)
        self.value_norm = nn.LayerNorm(value_input_dim)
        self.query_proj = nn.Linear(d_model, key_dim, bias=False)
        self.key_proj = nn.Linear(key_input_dim, key_dim, bias=False)
        self.value_proj = nn.Linear(value_input_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    @staticmethod
    def _validate_rank_and_last_dim(
        tensor: torch.Tensor, name: str, last_dim: int
    ) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [B, tokens, {last_dim}], got {tuple(tensor.shape)}")
        if tensor.shape[-1] != last_dim:
            raise ValueError(
                f"{name} must have last dimension {last_dim}, got {tensor.shape[-1]}"
            )

    def derive_static_key(self, key_input: torch.Tensor) -> torch.Tensor:
        """Project a calibration-derived static key once, returning ``[B,N,Dk]``."""
        self._validate_rank_and_last_dim(key_input, "key_input", self.key_input_dim)
        return self.key_proj(self.key_norm(key_input))

    def forward_cached(
        self,
        query: torch.Tensor,
        projected_key: torch.Tensor,
        value_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention without recomputing the static key projection."""
        self._validate_rank_and_last_dim(query, "query", self.d_model)
        self._validate_rank_and_last_dim(projected_key, "projected_key", self.key_dim)
        self._validate_rank_and_last_dim(value_input, "value_input", self.value_input_dim)
        if query.shape[0] != projected_key.shape[0] or query.shape[0] != value_input.shape[0]:
            raise ValueError("query, projected_key, and value_input must share batch size")
        if projected_key.shape[1] != value_input.shape[1]:
            raise ValueError("projected_key and value_input must share unit count N")
        if query.device != projected_key.device or query.device != value_input.device:
            raise ValueError("query, projected_key, and value_input must be on the same device")

        batch_size, num_queries, _ = query.shape
        num_units = projected_key.shape[1]
        query_norm = self.norm1(query)
        q = self.query_proj(query_norm).view(
            batch_size, num_queries, self.nhead, self.key_head_dim
        ).transpose(1, 2)
        k = projected_key.view(
            batch_size, num_units, self.nhead, self.key_head_dim
        ).transpose(1, 2)
        v = self.value_proj(self.value_norm(value_input)).view(
            batch_size, num_units, self.nhead, self.value_head_dim
        ).transpose(1, 2)

        scores_by_head = torch.matmul(q, k.transpose(-2, -1)) / (self.key_head_dim ** 0.5)
        attention_by_head = self.attn_dropout(torch.softmax(scores_by_head, dim=-1))
        attended = torch.matmul(attention_by_head, v).transpose(1, 2).reshape(
            batch_size, num_queries, self.value_dim
        )
        x = query + self.dropout(self.out_proj(attended))
        x = x + self.dropout(self.ffn(self.norm2(x)))

        # A stable, compact reporting tensor: no [B,N,N] neuron-neuron matrix.
        attention_scores = attention_by_head.mean(dim=1)
        return x, attention_scores

    def forward(
        self, query: torch.Tensor, key_input: torch.Tensor, value_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference non-cached path, useful for cache-equivalence tests."""
        return self.forward_cached(query, self.derive_static_key(key_input), value_input)


class CachedDecoupledMultiLayerCrossAttention(nn.Module):
    """Guarded cached decoupled transformer for the state-neutral one-layer pilot.

    The initial experiment is deliberately constrained to one decoder layer and
    ``Dk=32``.  Its persistent cache is therefore ``[B,N,32]``, smaller than the
    existing identity cache ``[B,N,50]``.  The explicit guard prevents callers
    from silently reusing the receipt after increasing layers or cache width.
    """

    def __init__(
        self,
        num_layers: int = 1,
        d_model: int = 128,
        nhead: int = 2,
        key_input_dim: int = 54,
        value_input_dim: int = 50,
        key_dim: int = 32,
        value_dim: int = 32,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cache_reference_width: int = 50,
        require_single_layer: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if cache_reference_width <= 0:
            raise ValueError("cache_reference_width must be positive")
        if require_single_layer and num_layers != 1:
            raise ValueError("The state-neutral decoupled pilot requires num_layers=1")
        if num_layers * key_dim > cache_reference_width:
            raise ValueError(
                "Projected static-key cache would exceed the reference identity cache: "
                f"num_layers*key_dim={num_layers * key_dim} > {cache_reference_width}"
            )

        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.key_input_dim = key_input_dim
        self.value_input_dim = value_input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.cache_reference_width = cache_reference_width
        self.require_single_layer = require_single_layer
        self.layers = nn.ModuleList([
            CachedDecoupledCrossAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                key_input_dim=key_input_dim,
                value_input_dim=value_input_dim,
                key_dim=key_dim,
                value_dim=value_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def derive_static_key(self, key_input: torch.Tensor) -> DecoupledKVState:
        """Create the only persistent session cache: one projected K per layer."""
        return DecoupledKVState(tuple(layer.derive_static_key(key_input) for layer in self.layers))

    def forward_cached(
        self,
        query: torch.Tensor,
        state: DecoupledKVState,
        value_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Decode from cached projected K; this path never invokes ``key_proj``."""
        if not isinstance(state, DecoupledKVState):
            raise TypeError("state must be a DecoupledKVState")
        if len(state.projected_keys) != self.num_layers:
            raise ValueError(
                f"state has {len(state.projected_keys)} projected keys, expected {self.num_layers}"
            )
        x = query
        attention_scores = []
        for layer, projected_key in zip(self.layers, state.projected_keys):
            x, score = layer.forward_cached(x, projected_key, value_input)
            attention_scores.append(score)
        return x, tuple(attention_scores)

    def forward(
        self, query: torch.Tensor, key_input: torch.Tensor, value_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Reference uncached decode; production streaming should use the cached API."""
        return self.forward_cached(query, self.derive_static_key(key_input), value_input)

    def cache_receipt(self, state: DecoupledKVState) -> Dict[str, object]:
        """Return a serializable receipt for the persistent static-key cache."""
        if not isinstance(state, DecoupledKVState):
            raise TypeError("state must be a DecoupledKVState")
        if len(state.projected_keys) != self.num_layers:
            raise ValueError("state layer count does not match this module")
        first = state.projected_keys[0]
        batch_size, num_units, key_width = first.shape
        if key_width != self.key_dim:
            raise ValueError("state key width does not match this module")
        tensor_receipts = []
        for index, projected_key in enumerate(state.projected_keys):
            if projected_key.shape[:2] != (batch_size, num_units):
                raise ValueError("all cached projected keys must have matching [B,N] shape")
            tensor_receipts.append({
                "name": "projected_static_key",
                "layer": index,
                "shape": list(projected_key.shape),
                "bytes": projected_key.numel() * projected_key.element_size(),
            })
        return {
            "schema_version": 1,
            "persistent_tensors": tensor_receipts,
            "cache_bytes": state.nbytes,
            "cache_bytes_formula": f"B*N*{self.num_layers}*{self.key_dim}*element_size",
            "reference_identity_width": self.cache_reference_width,
            "state_nonincreasing_vs_identity": self.num_layers * self.key_dim <= self.cache_reference_width,
            "excludes": [
                "identity",
                "direct_key_features",
                "raw_calibration",
                "values",
                "attention_scores",
            ],
        }

    def online_cost_receipt(self, batch_size: int, num_queries: int, num_units: int) -> Dict[str, object]:
        """MAC receipt showing that every unit-dependent term is linear in ``N``."""
        if batch_size <= 0 or num_queries <= 0 or num_units <= 0:
            raise ValueError("batch_size, num_queries, and num_units must all be positive")
        # Query and output projections are independent of N.  The other terms are
        # linear in N; there is deliberately no [N,N] attention operation.
        query_projection = batch_size * num_queries * self.d_model * self.key_dim * self.num_layers
        value_projection = batch_size * num_units * self.value_input_dim * self.value_dim * self.num_layers
        qk_scores = batch_size * num_queries * num_units * self.key_dim * self.num_layers
        weighted_values = batch_size * num_queries * num_units * self.value_dim * self.num_layers
        output_projection = batch_size * num_queries * self.value_dim * self.d_model * self.num_layers
        static_key_projection = batch_size * num_units * self.key_input_dim * self.key_dim * self.num_layers
        return {
            "schema_version": 1,
            "calibration_only_macs": {"static_key_projection": static_key_projection},
            "online_macs_per_frame": {
                "query_projection": query_projection,
                "value_projection": value_projection,
                "qk_scores": qk_scores,
                "weighted_values": weighted_values,
                "output_projection": output_projection,
                "unit_dependent_total": value_projection + qk_scores + weighted_values,
                "no_unit_quadratic_term": True,
            },
            "attention_score_shape": [batch_size, num_queries, num_units],
        }


class SpintModel(nn.Module):
    def __init__(self, model_dim, num_covariates, window_size,
                 num_heads=2, num_layers=1, num_id_layers=1,
                 use_learnable_id=True, learnable_id_type='mlp', learnable_rep=True,
                 dropout_rate=0.0, dynamic_dropout=False, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0, tf_drop_rate=0.1, readin_layer_type='mlp',
                ):
        super(SpintModel, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_id_layers = num_id_layers
        self.num_covariates = num_covariates
        self.window_size = window_size
        self.use_learnable_id = use_learnable_id
        self.learnable_id_type = learnable_id_type
        self.learnable_rep = learnable_rep
        self.dropout_rate = dropout_rate
        self.tf_drop_rate = tf_drop_rate
        self.dynamic_dropout = dynamic_dropout
        self.dynamic_dropout_low = dynamic_dropout_low
        self.dynamic_dropout_high = dynamic_dropout_high
        self.readin_layer_type = readin_layer_type

        self.fc_in = nn.Sequential(nn.Linear(window_size, model_dim),
                                   nn.ReLU(),
                                   nn.Linear(model_dim, model_dim))
        if self.use_learnable_id:
            if self.learnable_id_type == 'mlp':
                in_layers = [nn.LazyLinear(model_dim)]
                for _ in range(self.num_id_layers - 1):
                    in_layers.extend([nn.ReLU(), nn.Linear(model_dim, model_dim)])
                self.fc_id_in = nn.Sequential(*in_layers)
                out_layers = []
                for _ in range(self.num_id_layers - 1):
                    out_layers.extend([nn.Linear(model_dim, model_dim), nn.ReLU()])
                out_layers.append(nn.Linear(model_dim, window_size))
                self.fc_id_out = nn.Sequential(*out_layers)
            else:
                raise ValueError(f"Unsupported learnable ID type: {self.learnable_id_type}")
        else:
            raise ValueError(f"Non-learnable ID not supported. Must set use_learnable_id to True.")
                

        self.fc_out = nn.Linear(model_dim, window_size)
        if learnable_rep:       
            self.rep = nn.Parameter(torch.randn(1, num_covariates, window_size)) # 1xCxW
        else:
            self.rep = torch.arange(start=1, end=num_covariates+1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, window_size)/num_covariates # 1xCxW

        self.transformer = MultiLayerCrossAttention(num_layers=num_layers, d_model=model_dim, nhead=num_heads, dropout=self.tf_drop_rate)
        
    
    def forward(self, src, calib_trialized_neural_features=None):
        src = src.permute(0, 2, 1) # BxWxN -> BxNxW (batch size, num neurons, window size)
        
        batch_size = src.size(0)
        num_neurons = src.size(1)

        if self.use_learnable_id: 
            if self.learnable_id_type == 'mlp':
                id = calib_trialized_neural_features.permute(0,1,3,2) # BxMxTxN -> BxMxNxT
                id = self.fc_id_in(id) # BxMxNxT -> BxMxNxH
                id = torch.mean(id, dim=1, keepdim=False) # BxMxNxH -> BxNxH
                id = self.fc_id_out(id) # BxNxH -> BxNxW
            else:            
                raise ValueError(f"Unsupported learnable ID type: {self.learnable_id_type}")

            src = src + id # BxNxW + BxNxW -> BxNxW
        else:            
            raise ValueError(f"Non-learnable ID not supported. Must set use_learnable_id to True.")

        dropout_mask = torch.ones(batch_size, num_neurons).to(src) # BxN
        if self.dynamic_dropout:
            p = random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high)
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=self.training) # BxN
        else:
            dropout_mask = torch.nn.functional.dropout(dropout_mask, p=self.dropout_rate, training=self.training) # BxN
        src = src * dropout_mask.unsqueeze(-1) # BxNxW * (BxN -> BxNx1) -> BxNxW

        if self.readin_layer_type == 'mlp':
            src = self.fc_in(src) # BxNxW -> BxNxH
        else:
            raise ValueError(f"Unsupported readin_layer_type: {self.readin_layer_type}")

        rep = self.fc_in(self.rep).to(src) # 1xCxW -> 1xCxH

        transformer_output, _ = self.transformer(rep.repeat(src.size(0), 1, 1), src) # BxCxH

        output = self.fc_out(transformer_output) # BxCxH -> BxCxW
        behavior_pred = output.permute(0, 2, 1) # BxCxW -> BxWxC
        return behavior_pred
