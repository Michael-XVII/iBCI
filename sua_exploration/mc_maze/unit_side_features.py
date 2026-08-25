"""Per-unit side features for DANDI 000688 SUA: calibration-pool spike waveforms (F1/F2)
and calibration-pool directional tuning (T4/T8, E3_E4_ENCODER_PROGRAM.md section 1)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fcntl
import h5py
import numpy as np
from pynwb import NWBHDF5IO
from scipy.sparse.linalg import lsqr

from mc_maze.multisession_datamodule import (
    _cache_key,
    _exclusive_cache_lock,
    _source_fingerprint,
    _write_npz_atomically,
    calibration_pool_end_time,
    electrode_ids_from_units,
    list_datamodule_rewarded_trials,
    session_name_from_path,
)

logger = logging.getLogger(__name__)

# Existing SUA cache keys deliberately retain version 1/payload compatibility.
# Pseudo-MUA adds its own signal-view and electrode-mapping fields below, so it
# can never collide with a sorted-unit entry without forcing a SUA cache churn.
FEATURE_VERSION = 1
TEMPLATE_RIDGE_FEATURE_VERSION = 1
# ``t4c`` changed once after a train-only input audit, before any confidence-
# FiLM candidate was launched. Version 2 replaces the nearly duplicate
# covariance-area coordinate with the scale-free a/c uncertainty shape. Keep
# the global version at 1 so unrelated, already-validated waveform/T4 caches do
# not churn; cache payloads use ``feature_semantics_version`` below.
T4C_FEATURE_VERSION = 2
T4R_FEATURE_VERSION = 1
T4RQ_FEATURE_VERSION = 1
T4RQ_ANGULAR_EPS = 1.0e-8
T4RQ_ZERO_MODULATION_RELIABILITY = -20.0
T4R_POSTERIOR_FORMULA_VERSION = "isotropic_gaussian_posterior_mean_v1"
T4R_PRIOR_VARIANCE_FLOOR = 1.0e-8
# Selected by a fully train-only, leave-one-session-out audit.  This constant is
# frozen before any validation decoding run; it is not tuned per held-out
# session.  See results/sua_t4_confidence_shrinkage_audit_v1.
T4_WIENER_SHRINK_STRENGTH = 3.0
TEMPLATE_RIDGE_RIDGE = 1.0
TEMPLATE_RIDGE_LABEL_SHUFFLE_SEED = 20260813
WAVEFORM_SAMPLES = 48
REPOL_WINDOW = 10
NOISE_STD_EPS = 1e-6
TEMPLATE_MAX_EPS = 1e-6
TRAIN_CLIP_QUANTILES = (0.01, 0.99)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "f1": ("p2p", "noise_std", "snr"),
    "f2": ("p2p", "noise_std", "snr", "pt_width", "pt_ratio", "repol_slope"),
}

# E3 directional tuning features (E3_E4_ENCODER_PROGRAM.md section 1.2): computed from the
# calibration-pool trials' per-direction mean firing rates, not from waveforms, so they are
# kept in a separate registry from FEATURE_GROUPS above (whose values are the scalar-name
# tuples consumed by _scalar_features_from_template -- a waveform-specific mechanism that
# does not apply here). "t4" is the cosine-tuning fit [m*cos(phi), m*sin(phi), m, b]; "t8" is
# the raw per-direction mean-rate vector. Both use the same fixed canonical direction order
# (CANONICAL_DIRECTIONS_RAD below) so column k means the same physical direction in every
# session.
TUNING_NUM_DIRECTIONS = 8
# Fixed, session-independent center-out target directions (radians), ascending, matching the
# (-pi, pi] convention numpy's arctan2 and this dataset's target_dir column both use (verified
# 2026-07-25 against sub-C ses-CO-20151103: observed unique target_dir values are exactly
# these 8, 45 degrees apart). This order is the canonical t8 column order for every session --
# it is derived from the fixed task geometry, never fit per session, so it cannot drift.
CANONICAL_DIRECTIONS_RAD: tuple[float, ...] = tuple(
    -3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(TUNING_NUM_DIRECTIONS)
)
TUNING_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "t4": ("m_cos_phi", "m_sin_phi", "m", "b"),
    "t4r": ("posterior_m_cos_phi", "posterior_m_sin_phi", "posterior_m", "b"),
    "t4rq": ("posterior_m_cos_phi", "posterior_m_sin_phi", "posterior_m", "b", "angular_reliability"),
    # E04 reuses E03's carrier but routes q_theta only to attention logits.
    "t4rql": ("posterior_m_cos_phi", "posterior_m_sin_phi", "posterior_m", "b", "angular_reliability"),
    "t4w3": (
        "wiener3_m_cos_phi",
        "wiener3_m_sin_phi",
        "wiener3_m",
        "b",
    ),
    "t4c": (
        "m_cos_phi", "m_sin_phi", "m", "b",
        "log_residual_variance", "c_shape_log_condition_cac",
    ),
    "t8": tuple(f"dir_{k}" for k in range(TUNING_NUM_DIRECTIONS)),
}
TEMPLATE_RIDGE_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "tr4": ("template_cos_weight", "template_sin_weight", "template_norm", "support_rate"),
    "trls4": (
        "label_shuffled_template_cos_weight",
        "label_shuffled_template_sin_weight",
        "label_shuffled_template_norm",
        "support_rate",
    ),
}
# Modulation depth m = hypot(a, c) below which a unit's cosine tuning fit is treated as flat
# (no detectable direction preference). Rates are in Hz; this mirrors the NOISE_STD_EPS /
# TEMPLATE_MAX_EPS "numerically zero" convention already used for the waveform features above
# rather than a fraction of firing rate, since m=0 is a meaningful, not just numerically
# fragile, degenerate value here (a genuinely untuned unit).
MODULATION_EPS = 1e-6

SIDE_FEATURE_DIMS: dict[str, int] = {
    "none": 0,
    "f1": 3,
    "f2": 6,
    "f3": 6,
    "fs1": 3,
    "fs2": 6,
    "fs3": 6,
    "t4": 4,
    "t4r": 4,
    "t4rq": 5,
    "t4rql": 5,
    "t4w3": 4,
    "t8": 8,
    "ts4": 4,
    "ts4r": 4,
    "ts4w3": 4,
    "ts8": 8,
    # T4-substrate electrode designs (docs/ELECTRODE_ANCHOR_DESIGNS.md). All three reuse T4's
    # own 4-dim cosine-tuning fit as the continuous side_dim concatenated at the psi input;
    # none of them widen that concat beyond T4 except "t4e" (design A), whose extra 8 dims
    # come from ELECTRODE_EMBED_DIM via uses_electrode_embedding()/post_pool_side_dim() below,
    # not from this table.
    "t4e": 4,
    "t4e_shuffled": 4,
    "t4gate": 4,
    "t4gate_shuffled": 4,
    "t4anchor": 4,
    "t4anchor_shuffled": 4,
    # Stage-0 same-electrode relation. These retain the already-validated T4
    # values; the new content is equality-based membership, not an absolute
    # electrode lookup or a retry of waveform/SNR concat.
    "t4rel": 4,
    "t4rel_membership_shuffled": 4,
    "t4rel_nogroup": 4,
    # Cached [T4(4), fit-confidence(2)] from the same first-M_T4 support.
    "t4cf": 6,
    "t4cf_ts4": 6,
    "t4cf_confidence_shuffled": 6,
    "t4cf_residual": 6,
    "t4cf_residual_shuffled": 6,
    "tr4": 4,
    "trs4": 4,
    "trls4": 4,
    "trz4": 4,
}

# Learned electrode-index embedding width for F3 (UNIT_SIDE_FEATURE_ABLATION.md section 6)
# and for design A ("t4e", docs/ELECTRODE_ANCHOR_DESIGNS.md).
ELECTRODE_EMBED_DIM = 8

# Every feature-group token this module knows how to compute (real groups only; shuffled
# controls resolve to one of these via base_feature_group before compute_unit_side_features_
# uncached / fit_side_feature_stats / load_unit_side_features ever see them). Used for the
# "is this a known feature_group" validity checks below instead of FEATURE_GROUPS alone,
# since FEATURE_GROUPS is deliberately waveform-only (see TUNING_FEATURE_NAMES above).
# t4e/t4gate/t4anchor are NOT added here: base_feature_group() resolves all three straight to
# "t4" (already a KNOWN_FEATURE_GROUPS member via TUNING_FEATURE_NAMES), exactly the same
# indirection "f3" uses to reach "f2" -- see the "f3" test in test_side_feature_encoder.py's
# test_known_feature_groups_covers_waveform_and_tuning_real_groups, which asserts fs1/fs3/ts4
# (any group that only ever appears post-resolution) are absent from this set.
KNOWN_FEATURE_GROUPS: frozenset[str] = (
    frozenset(FEATURE_GROUPS)
    | frozenset(TUNING_FEATURE_NAMES)
    | frozenset(TEMPLATE_RIDGE_FEATURE_NAMES)
    | frozenset({"f3"})
)

# Dimension-matched shuffled controls (UNIT_SIDE_FEATURE_ABLATION.md section 6, revised
# 2026-07-25): FS1 permutes the same 3-dim F1 feature set along the unit axis; FS2 permutes
# the same 6-dim F2 feature set. The original single 6-dim "fs" compared a 3-dim F1 against
# a 6-dim control -- i.e. two different post_pool architectures (fan_in 67 vs 70) whose RNG
# streams diverge from the first layer on -- which the charter now documents as a defect
# (F1-FS confounded side_dim with feature content). Every content gate must only ever
# compare a feature group against its own dimension-matched control. TS4/TS8 extend the same
# pattern to the E3 tuning features (E3_E4_ENCODER_PROGRAM.md section 1.2): TS4 permutes T4
# along the unit axis, TS8 permutes T8. t4e_shuffled/t4gate_shuffled/t4anchor_shuffled extend
# it again to the T4-substrate electrode designs (docs/ELECTRODE_ANCHOR_DESIGNS.md): each
# permutes only the ELECTRODE ID assignment along the unit axis (never the T4 tuning values,
# which stay correctly attached to their real unit) -- the same electrode-shuffle pattern FS3
# established, not the feature-shuffle pattern FS1/FS2/TS4/TS8 use. See
# is_electrode_shuffle_control below.
SHUFFLED_CONTROL_BASE_FEATURE_GROUP: dict[str, str] = {
    "fs1": "f1",
    "fs2": "f2",
    "fs3": "f3",
    "ts4": "t4",
    "ts4r": "t4r",
    "ts4w3": "t4w3",
    "ts8": "t8",
    "t4e_shuffled": "t4e",
    "t4gate_shuffled": "t4gate",
    "t4anchor_shuffled": "t4anchor",
    "t4rel_membership_shuffled": "t4rel",
    "trs4": "tr4",
}

# Group tokens whose electrode mechanism is a learned CONCAT embedding at the psi input (F3's
# mechanism, reused unchanged by design A / "t4e"). Narrower than uses_electrode_ids() below:
# this set gates post_pool_side_dim()'s +ELECTRODE_EMBED_DIM and train_variant_dandi688.py's
# electrode_embed_dim -- i.e. "does the SideFeatureEarlyPoolEncoder base class itself need to
# build an nn.Embedding and widen post_pool's input". Design D/C ("t4gate"/"t4anchor") do NOT
# belong here: they apply their own, separately-parameterized electrode mechanism to psi's
# OUTPUT (see ElectrodeGateEarlyPoolEncoder/ElectrodeAnchorEarlyPoolEncoder in
# streaming_encoders.py), not a concat at its input, so they must not also widen post_pool.
_ELECTRODE_EMBED_CONCAT_GROUPS: frozenset[str] = frozenset({"f3", "fs3", "t4e", "t4e_shuffled"})

# Group tokens that need per-unit electrode ids attached to the dataset at all (superset of
# _ELECTRODE_EMBED_CONCAT_GROUPS above): every T4-substrate electrode design (A/D/C) and F3/FS3
# need electrode_ids loaded and num_electrodes computed, regardless of which mechanism (concat
# embedding, multiplicative gate, or additive anchor) actually consumes them downstream. This
# is the gate multisession_datamodule.py / eval_adaptation_dandi688.py / train_variant_dandi688.py
# (the num_electrodes branch) must use; uses_electrode_embedding() remains the narrower gate for
# the concat-specific wiring described above.
_ELECTRODE_ID_GROUPS: frozenset[str] = _ELECTRODE_EMBED_CONCAT_GROUPS | frozenset(
    {"t4gate", "t4gate_shuffled", "t4anchor", "t4anchor_shuffled",
     "t4rel", "t4rel_membership_shuffled"}
)

# Electrode-shuffle (not feature-shuffle) controls: permute electrode id assignment only.
_ELECTRODE_SHUFFLE_CONTROLS: frozenset[str] = frozenset(
    {"fs3", "t4e_shuffled", "t4gate_shuffled", "t4anchor_shuffled",
     "t4rel_membership_shuffled"}
)


def is_shuffled_control(side_feature_group: str) -> bool:
    """True for permuted-control tokens (fs1, fs2, fs3, ts4, ts8, t4e_shuffled,
    t4gate_shuffled, t4anchor_shuffled)."""
    return side_feature_group in SHUFFLED_CONTROL_BASE_FEATURE_GROUP


def uses_electrode_embedding(side_feature_group: str) -> bool:
    """True when the encoder must build a learned electrode-index CONCAT embedding at the
    psi input (F3's mechanism: f3/fs3, and design A's t4e/t4e_shuffled). False for design
    D/C (t4gate/t4anchor and their shuffled controls), whose electrode mechanism is NOT a
    concat -- see uses_electrode_ids() for the broader "needs electrode ids at all" gate
    those designs (and this function's own groups) share.
    """
    return side_feature_group in _ELECTRODE_EMBED_CONCAT_GROUPS


def uses_electrode_ids(side_feature_group: str) -> bool:
    """True when per-unit electrode ids (and num_electrodes) must be attached/computed at
    all, regardless of mechanism: F3/FS3 and every T4-substrate electrode design (A: t4e /
    D: t4gate / C: t4anchor, plus their shuffled controls). Superset of
    uses_electrode_embedding(); data-loading call sites (multisession_datamodule.py,
    eval_adaptation_dandi688.py, train_variant_dandi688.py's num_electrodes branch) must use
    this, not the narrower uses_electrode_embedding(), or D/C's own electrode tables would
    never receive electrode ids / a correctly-sized vocabulary.
    """
    return side_feature_group in _ELECTRODE_ID_GROUPS


def is_electrode_shuffle_control(side_feature_group: str) -> bool:
    """True for FS3 and the T4-substrate electrode shuffled controls (t4e_shuffled,
    t4gate_shuffled, t4anchor_shuffled): permute electrode ids along the unit axis, never
    the continuous feature values (waveform scalars or T4 tuning) themselves."""
    return side_feature_group in _ELECTRODE_SHUFFLE_CONTROLS


def uses_electrode_relation_membership(side_feature_group: str) -> bool:
    """True for Stage-0 equality-only same-electrode relation tokens.

    Unlike ``uses_electrode_ids``, this does not authorize an absolute-ID table:
    callers may use IDs solely for equality-based segmented groups.  In
    pseudo-MUA view each physical channel is necessarily a singleton group.
    """
    return side_feature_group in {"t4rel", "t4rel_membership_shuffled"}


def is_feature_shuffle_control(side_feature_group: str) -> bool:
    """True when continuous side features are permuted along the unit axis (fs1/fs2/ts4/ts8)."""
    return is_shuffled_control(side_feature_group) and not is_electrode_shuffle_control(
        side_feature_group
    )


def confidence_component_shuffle(side_feature_group: str) -> str | None:
    """Return the cached t4c component shuffled by a confidence-FiLM control.

    These controls must not use the generic row shuffle: ``t4cf_ts4`` moves
    only T4 (columns 0:4), while ``t4cf_confidence_shuffled`` moves only C
    (columns 4:6), preserving the other component's unit alignment.
    """
    if side_feature_group == "t4cf_ts4":
        return "t4"
    if side_feature_group == "t4cf_confidence_shuffled":
        return "confidence"
    if side_feature_group == "t4cf_residual_shuffled":
        return "residual"
    return None


def is_template_ridge_zero_control(side_feature_group: str) -> bool:
    """True for the exact-zero Template-Ridge floor control."""
    return side_feature_group == "trz4"


def permute_t4c_component(features: np.ndarray, *, component: str, permutation_seed: int) -> np.ndarray:
    """Permute one component along units while retaining every component marginal."""
    if features.ndim != 2 or features.shape[1] != 6:
        raise ValueError(f"t4c features must be [units,6], got {features.shape}")
    if component not in {"t4", "confidence", "residual", "geometry"}:
        raise ValueError(
            "component must be 't4', 'confidence', 'residual', or 'geometry'"
        )
    result = features.copy()
    columns = {
        "t4": slice(0, 4),
        "confidence": slice(4, 6),
        "residual": slice(4, 5),
        "geometry": slice(5, 6),
    }[component]
    result[:, columns] = features[np.random.RandomState(permutation_seed).permutation(features.shape[0]), columns]
    return result


def base_feature_group(side_feature_group: str) -> str:
    """The waveform/tuning registry key to compute and z-score for ``side_feature_group``.

    Identity for "f1"/"f2"/"t4"/"t8" (and for anything else, e.g. "none"); resolves shuffled
    controls to the real feature set they permute ("fs1" -> "f1", "fs2" -> "f2"). F3/FS3
    reuse the F2 waveform scalars; only the electrode assignment (and FS3's permutation of
    it) differs. The T4-substrate electrode designs (t4e/t4gate/t4anchor and their shuffled
    controls, docs/ELECTRODE_ANCHOR_DESIGNS.md) all resolve to "t4" the same way -- every one
    of them reuses T4's own cosine-tuning fit unchanged; only the electrode mechanism differs
    between them.
    """
    group = SHUFFLED_CONTROL_BASE_FEATURE_GROUP.get(side_feature_group, side_feature_group)
    if group == "f3":
        return "f2"
    if group in {
        "t4e", "t4gate", "t4anchor", "t4rel", "t4rel_nogroup",
        "t4cf", "t4cf_ts4", "t4cf_confidence_shuffled",
        "t4cf_residual", "t4cf_residual_shuffled",
    }:
        return "t4c" if group.startswith("t4cf") else "t4"
    if group == "trz4":
        return "tr4"
    return group


def side_features_use_behavior_labels(side_feature_group: str) -> bool:
    """Whether a side-feature token derives from rewarded trial behavior labels.

    This is intentionally defined from the resolved feature substrate rather
    than maintained as a second, hand-written token list in each evaluator.
    Tuning groups consume rewarded trial direction/rate support; shuffled and
    residual controls retain that same label-derived substrate.  Waveform-only
    groups and ``none`` do not.
    """

    resolved = base_feature_group(side_feature_group)
    return resolved in TUNING_FEATURE_NAMES or resolved in TEMPLATE_RIDGE_FEATURE_NAMES


def feature_semantics_version(side_feature_group: str) -> int:
    """Semantic/cache version for the resolved continuous feature group."""
    resolved = base_feature_group(side_feature_group)
    return (
        T4C_FEATURE_VERSION
        if resolved == "t4c"
        else T4RQ_FEATURE_VERSION
        if resolved in {"t4rq", "t4rql"}
        else T4R_FEATURE_VERSION
        if resolved == "t4r"
        else TEMPLATE_RIDGE_FEATURE_VERSION
        if resolved in TEMPLATE_RIDGE_FEATURE_NAMES
        else FEATURE_VERSION
    )


def post_pool_side_dim(side_feature_group: str) -> int:
    """Total width concat to pooled hidden features before ``post_pool`` (continuous + embed).

    Only meaningful for groups whose electrode mechanism IS a concat embedding
    (uses_electrode_embedding()); design D/C add ELECTRODE_EMBED_DIM to nothing -- their
    electrode tables are separate parameters applied to psi's OUTPUT, not extra post_pool
    input columns, so this function correctly returns just SIDE_FEATURE_DIMS for them (4, T4's
    own width) rather than double-counting an embedding they do not concatenate.
    """
    dim = SIDE_FEATURE_DIMS.get(side_feature_group, 0)
    if uses_electrode_embedding(side_feature_group):
        return dim + ELECTRODE_EMBED_DIM
    return dim


@dataclass(frozen=True)
class SideFeatureMetadata:
    feature_group: str
    feature_version: int
    pool_size: int
    cache_key: str
    degenerate_unit_count: int
    zero_spike_unit_count: int
    single_spike_unit_count: int
    zero_noise_std_unit_count: int
    zero_template_max_unit_count: int
    # E3 tuning-only degenerate reasons (0 for waveform feature groups f1/f2). Defaulted so
    # every existing keyword-argument call site (all of which predate these fields) keeps
    # working unchanged.
    zero_modulation_unit_count: int = 0
    insufficient_direction_unit_count: int = 0
    template_ridge_constructed_rows: int = 0
    template_ridge_feature_count: int = 0
    template_ridge_condition: float = 0.0
    template_ridge_trace_hat: float = 0.0
    template_ridge_profile_sha256: str = ""
    template_ridge_alignment_event: str = ""
    posterior_prior_variance: float = 0.0
    posterior_prior_sha256: str = ""
    posterior_design_rank: int = 0
    posterior_design_condition: float = 0.0
    posterior_reliability_formula: str = ""
    posterior_reliability_epsilon: float = 0.0
    posterior_reliability_zero_floor: float = 0.0


def _pool_context_key(
    *,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
) -> dict[str, int | str]:
    return {
        "bin_size_ms": bin_size_ms,
        "window_size": window_size,
        "trial_result_filter": trial_result_filter,
    }


def _validate_signal_view(signal_view: str) -> None:
    if signal_view not in {"sua", "pseudo_mua"}:
        raise ValueError(
            "signal_view must be one of {'sua', 'pseudo_mua'}; "
            f"got {signal_view!r}"
        )


def _electrode_mapping_fingerprint(nwb_path: Path) -> str:
    """Stable hash of the sorted-unit -> NWB-electrode mapping for cache identity.

    A source-file fingerprint is normally sufficient, but this explicit mapping
    component documents and enforces the semantic dependency of pseudo-MUA T4:
    changing which sorted units belong to an electrode must never reuse the old
    channel-level tuning matrix or train normalization statistics.
    """
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        electrode_ids = electrode_ids_from_units(nwb.units.to_dataframe())
    return hashlib.sha256(electrode_ids.tobytes()).hexdigest()


def pool_trial_rates_by_electrode(
    trial_rates: np.ndarray,
    electrode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum per-sorted-unit trial rates into deterministic electrode channels.

    ``trial_rates`` has shape ``[units, trials]``.  Rates may be summed because
    every unit's count is divided by the same trial duration.  This is purposely
    not an average of fitted/unit T4 rows: the aggregate channel rate for every
    pool trial is formed first, and the existing directional cosine fit then sees
    exactly the pseudo-MUA signal presented to the model.
    """
    if trial_rates.ndim != 2:
        raise ValueError(f"Expected trial rates [units, trials], got {trial_rates.shape}")
    if electrode_ids.ndim != 1 or electrode_ids.shape[0] != trial_rates.shape[0]:
        raise ValueError(
            "electrode_ids must contain exactly one value per unit; got "
            f"{electrode_ids.shape} for {trial_rates.shape[0]} units"
        )
    channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
    pooled = np.zeros((channel_ids.size, trial_rates.shape[1]), dtype=np.float64)
    np.add.at(pooled, inverse, trial_rates)
    return pooled, channel_ids


def permute_side_feature_rows(features: np.ndarray, *, permutation_seed: int) -> np.ndarray:
    """Deterministically permute only the channel/unit axis of a side matrix."""
    if features.ndim != 2:
        raise ValueError(f"Expected side features [rows, dims], got {features.shape}")
    generator = np.random.RandomState(permutation_seed)
    return features[generator.permutation(features.shape[0])]


def _side_stats_cache_path(
    cache_dir: Path,
    train_files: Sequence[Path],
    *,
    feature_group: str,
    pool_size: int,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
    signal_view: str = "sua",
    template_profile_hash: str | None = None,
    posterior_prior_hash: str | None = None,
) -> Path:
    payload = {
        "cache_format_version": feature_semantics_version(feature_group),
        "kind": "side_feature_stats",
        "feature_group": feature_group,
        "pool_size": pool_size,
        **_pool_context_key(
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        ),
        "train_sources": [_source_fingerprint(path) for path in train_files],
    }
    if signal_view == "pseudo_mua":
        payload["signal_view"] = signal_view
        payload["train_electrode_mappings"] = [
            _electrode_mapping_fingerprint(path) for path in train_files
        ]
    if template_profile_hash is not None:
        payload["template_profile_sha256"] = template_profile_hash
    if posterior_prior_hash is not None:
        payload["posterior_prior_sha256"] = posterior_prior_hash
    return cache_dir / "side_feature_stats" / f"{_cache_key(payload)[:20]}.npz"


def _side_feature_cache_path(
    cache_dir: Path,
    nwb_path: Path,
    *,
    feature_group: str,
    pool_size: int,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
    signal_view: str = "sua",
    template_profile_hash: str | None = None,
    posterior_prior_hash: str | None = None,
) -> Path:
    payload = {
        "cache_format_version": feature_semantics_version(feature_group),
        "kind": "unit_side_features",
        "feature_group": feature_group,
        "pool_size": pool_size,
        **_pool_context_key(
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        ),
        "source": _source_fingerprint(nwb_path),
    }
    if signal_view == "pseudo_mua":
        payload["signal_view"] = signal_view
        payload["electrode_mapping"] = _electrode_mapping_fingerprint(nwb_path)
    if template_profile_hash is not None:
        payload["template_profile_sha256"] = template_profile_hash
    if posterior_prior_hash is not None:
        payload["posterior_prior_sha256"] = posterior_prior_hash
    key = _cache_key(payload)[:20]
    return cache_dir / "side_features" / f"{session_name_from_path(nwb_path)}_{key}.npz"


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    slope, _ = np.polyfit(x, values.astype(np.float64), 1)
    return float(slope)


def _scalar_features_from_template(
    template: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    p2p = float(template.max() - template.min())
    noise_std = float(residuals.std()) if residuals.size else 0.0
    if noise_std <= NOISE_STD_EPS:
        snr = 0.0
    else:
        snr = p2p / noise_std
    argmax = int(template.argmax())
    argmin = int(template.argmin())
    pt_width = float(abs(argmax - argmin))
    template_max = float(template.max())
    template_min = float(template.min())
    if abs(template_max) <= TEMPLATE_MAX_EPS:
        pt_ratio = 0.0
    else:
        pt_ratio = float(abs(template_min) / abs(template_max))
    trough = argmin
    window_end = min(trough + REPOL_WINDOW, template.size)
    repol_segment = template[trough:window_end]
    repol_slope = _linear_slope(repol_segment)
    return {
        "p2p": p2p,
        "noise_std": noise_std,
        "snr": snr,
        "pt_width": pt_width,
        "pt_ratio": pt_ratio,
        "repol_slope": repol_slope,
    }


def _unit_spike_bounds(waveforms_index_index: np.ndarray, unit_idx: int) -> tuple[int, int]:
    start = 0 if unit_idx == 0 else int(waveforms_index_index[unit_idx - 1])
    end = int(waveforms_index_index[unit_idx])
    return start, end


def _read_unit_waveform_block(
    waveforms: h5py.Dataset,
    waveforms_index: h5py.Dataset,
    spike_start: int,
    num_spikes: int,
) -> np.ndarray:
    """Read ``num_spikes`` consecutive waveforms for one unit in a single h5py slice."""
    if num_spikes <= 0:
        raise ValueError("num_spikes must be positive")
    first_global = spike_start
    last_global = spike_start + num_spikes - 1
    block_start = int(waveforms_index[first_global]) - WAVEFORM_SAMPLES
    block_end = int(waveforms_index[last_global])
    block = waveforms[block_start:block_end, 0].astype(np.float32)
    expected = num_spikes * WAVEFORM_SAMPLES
    if block.size != expected:
        raise ValueError(
            f"Expected {expected} waveform samples for {num_spikes} spikes, got {block.size}"
        )
    return block.reshape(num_spikes, WAVEFORM_SAMPLES)


def _in_pool_spike_prefix(
    spike_times: np.ndarray,
    pool_end_time: float,
) -> tuple[int, np.ndarray]:
    """Return the count and boolean mask for the in-pool prefix of a unit spike train."""
    if spike_times.size and not np.all(spike_times[:-1] <= spike_times[1:]):
        raise ValueError("Unit spike_times must be non-decreasing")
    in_pool = spike_times <= pool_end_time
    if not np.any(in_pool):
        return 0, in_pool
    last_in_pool = int(np.where(in_pool)[0][-1])
    if not np.all(in_pool[: last_in_pool + 1]):
        raise ValueError("In-pool spikes must form a prefix of the unit spike train")
    return last_in_pool + 1, in_pool


def _nearest_canonical_direction_index(target_dir_rad: float) -> int:
    """Snap a raw ``target_dir`` angle (radians) to the closest of the 8
    ``CANONICAL_DIRECTIONS_RAD``, using circular (mod 2*pi) distance.

    Circular distance (rather than plain subtraction) makes this robust to any sign/wrap
    convention a session might use for the boundary angle (+pi vs -pi are the same physical
    direction), and to any sub-ULP floating differences -- it does not rely on bit-identical
    values across sessions.
    """
    directions = np.asarray(CANONICAL_DIRECTIONS_RAD, dtype=np.float64)
    wrapped = (directions - target_dir_rad + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))


def _fit_cosine_tuning(directions_rad: np.ndarray, mean_rates: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares fit of ``rate(theta) = b + a*cos(theta) + c*sin(theta)``, which is the
    same model as ``b + m*cos(theta - phi)`` with ``a = m*cos(phi)``, ``c = m*sin(phi)``.

    Returns ``(a, c, m, b)`` -- i.e. exactly the four T4 dimensions in order, so callers never
    need ``phi`` itself. ``directions_rad``/``mean_rates`` must be the same length and hold one
    row per *distinct* direction actually observed in the calibration pool (not one row per
    trial): fitting the per-direction means, not the raw per-trial rates, matches
    E3_E4_ENCODER_PROGRAM.md section 1.2 ("fit ... over the pool trials' mean firing rates per
    direction") and does not implicitly reweight directions by how many pool trials happened to
    land on them.

    Requires only >=1 rows to run without raising (``np.linalg.lstsq`` returns the minimum-norm
    solution for an underdetermined system); callers must apply the
    ``len(present_directions) < 2`` degeneracy gate themselves (E3_E4_ENCODER_PROGRAM.md section
    1.4) before calling this, since that gate is a session-wide fact shared by every unit, not a
    per-fit numerical failure this function would detect.
    """
    design = np.stack(
        [np.ones_like(directions_rad), np.cos(directions_rad), np.sin(directions_rad)], axis=1
    )
    coefficients, *_ = np.linalg.lstsq(design, mean_rates, rcond=None)
    b, a, c = (float(value) for value in coefficients)
    m = float(math.hypot(a, c))
    return a, c, m, b


def tuning_fit_confidence_descriptor(
    trial_rates: np.ndarray,
    direction_indices: np.ndarray,
    *,
    selected_t4: np.ndarray | None = None,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Fit-derived ``[log residual variance, 0.5 log condition(C_ac)]``.

    Uses valid labelled *trial-level* rates from the first-M_T4 support and
    ``X=[1, cos(theta), sin(theta)]``. Residuals are evaluated against the
    actual selected T4 coefficients (the existing equal-per-direction-mean
    cosine fit), not against a second trial-weighted refit. Thus this measures
    uncertainty of the T4 value the encoder really receives even when the
    chronological support is direction-imbalanced.

    ``C=sigma²(X'X)^-1`` and ``C_ac`` is its a/c block.  The first coordinate
    measures unit-specific fit noise.  The condition-number coordinate is
    scale-free, so it measures only session-level directional-design
    anisotropy rather than repeating residual variance.  A train-only audit
    rejected the former covariance-area coordinate because it correlated
    0.975 with log residual variance.
    """
    directions = np.asarray(direction_indices, dtype=np.int64)
    valid = directions >= 0
    valid_directions = directions[valid]
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in valid_directions]
    )
    response = np.asarray(trial_rates, dtype=np.float64)[valid]
    if response.size < 3:
        raise ValueError("t4c confidence requires at least three valid labelled trials")
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else math.inf
    if rank != 3 or not math.isfinite(condition):
        raise ValueError(
            f"t4c confidence requires rank=3 finite-condition design; got rank={rank}, condition={condition}"
        )
    if selected_t4 is None:
        present_directions = sorted({int(index) for index in valid_directions})
        direction_theta = np.asarray(
            [CANONICAL_DIRECTIONS_RAD[index] for index in present_directions],
            dtype=np.float64,
        )
        direction_means = np.asarray(
            [
                response[valid_directions == index].mean()
                for index in present_directions
            ],
            dtype=np.float64,
        )
        a, c, _m, b = _fit_cosine_tuning(direction_theta, direction_means)
    else:
        selected = np.asarray(selected_t4, dtype=np.float64)
        if selected.shape != (4,) or not np.isfinite(selected).all():
            raise ValueError(
                f"selected_t4 must be one finite four-vector, got {selected.shape}"
            )
        a, c, _m, b = (float(value) for value in selected)
    beta = np.asarray([b, a, c], dtype=np.float64)
    residual = response - design @ beta
    residual_variance = float(np.dot(residual, residual) / max(1, response.size - 3))
    # The scale factor sigma² cancels from condition(C_ac), including for a
    # zero-residual unit. Compute it from the design covariance directly to
    # avoid an undefined condition number for an all-zero covariance matrix.
    design_c_ac = np.linalg.inv(design.T @ design)[1:3, 1:3]
    c_shape = 0.5 * math.log(max(float(np.linalg.cond(design_c_ac)), 1.0))
    return np.asarray(
        [math.log(residual_variance + eps), c_shape], dtype=np.float32
    )


def uncertainty_wiener_shrink_t4(
    t4: np.ndarray,
    log_residual_variance: np.ndarray,
    direction_indices: np.ndarray,
    *,
    strength: float = T4_WIENER_SHRINK_STRENGTH,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink T4's ``a,c`` using fit uncertainty; keep ``b`` unchanged.

    For unit ``i``, signal power is ``a_i²+c_i²`` and uncertainty power is
    ``sigma_i² trace([(X'X)^-1]_{a,c})`` from the same labelled support.  The
    frozen Wiener factor is ``signal/(signal + strength*uncertainty)``.
    """
    features = np.asarray(t4, dtype=np.float64)
    log_variance = np.asarray(log_residual_variance, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError(f"t4 must be [units,4], got {features.shape}")
    if log_variance.shape != (features.shape[0],):
        raise ValueError(
            "log_residual_variance must have one value per T4 unit"
        )
    if (
        not np.isfinite(features).all()
        or not np.isfinite(log_variance).all()
        or not math.isfinite(strength)
        or strength < 0.0
    ):
        raise ValueError("T4 shrinkage inputs and strength must be finite/nonnegative")
    directions = np.asarray(direction_indices, dtype=np.int64)
    valid = directions >= 0
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions[valid]],
        dtype=np.float64,
    )
    design = np.stack(
        [np.ones_like(theta), np.cos(theta), np.sin(theta)],
        axis=1,
    )
    if int(np.linalg.matrix_rank(design)) != 3:
        raise ValueError("T4 shrinkage requires a rank-3 labelled direction design")
    covariance_trace = float(
        np.trace(np.linalg.inv(design.T @ design)[1:3, 1:3])
    )
    signal = np.square(features[:, 0]) + np.square(features[:, 1])
    uncertainty = np.exp(log_variance) * covariance_trace
    factors = signal / (signal + strength * uncertainty + eps)
    factors = np.clip(factors, 0.0, 1.0)
    shrunk = features.copy()
    shrunk[:, :2] *= factors[:, None]
    shrunk[:, 2] = np.hypot(shrunk[:, 0], shrunk[:, 1])
    return shrunk.astype(np.float32), factors.astype(np.float32)



def _posterior_design(direction_indices: np.ndarray) -> tuple[np.ndarray, int, float]:
    directions = np.asarray(direction_indices, dtype=np.int64)
    valid = directions >= 0
    theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions[valid]], dtype=np.float64)
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else math.inf
    if rank != 3 or not math.isfinite(condition):
        raise ValueError(f"t4r posterior requires rank=3 finite-condition design; got rank={rank}, condition={condition}")
    return design, rank, condition


def _trial_level_t4_ols(trial_rates: np.ndarray, direction_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    rates = np.asarray(trial_rates, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int64)
    if rates.ndim != 2 or rates.shape[1] != directions.size:
        raise ValueError("trial_rates must be [units,trials] aligned to direction_indices")
    design, rank, condition = _posterior_design(directions)
    response = rates[:, directions >= 0]
    coefficients = np.linalg.lstsq(design, response.T, rcond=None)[0].T
    residual = response - coefficients @ design.T
    dof = response.shape[1] - rank
    if dof <= 0:
        raise ValueError("t4r posterior requires more labelled trials than design rank")
    variance = np.sum(np.square(residual), axis=1) / float(dof)
    if not np.isfinite(coefficients).all() or not np.isfinite(variance).all():
        raise ValueError("t4r OLS statistics are non-finite")
    return coefficients, variance, design, rank, condition


def _t4r_prior_hash(receipt: dict[str, object]) -> str:
    payload = {key: receipt[key] for key in ("formula_version", "prior_variance", "pool_size", "source_sessions", "source_fingerprints", "source_unit_count")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_t4r_posterior_prior(train_files: Sequence[Path], *, pool_size: int, bin_size_ms: int, window_size: int, trial_result_filter: str, signal_view: str = "sua") -> dict[str, object]:
    """Estimate the isotropic direction prior from source-train sessions only."""
    _validate_signal_view(signal_view)
    beta_chunks: list[np.ndarray] = []
    noise_trace_chunks: list[np.ndarray] = []
    for nwb_path in train_files:
        trials = list_datamodule_rewarded_trials(nwb_path, bin_size_ms=bin_size_ms, window_size=window_size, trial_result_filter=trial_result_filter)[:pool_size]
        if len(trials) < pool_size:
            raise ValueError(f"{nwb_path}: fewer than {pool_size} rewarded source trials")
        directions = np.asarray([_nearest_canonical_direction_index(trial["target_dir"]) if trial.get("target_dir") is not None else -1 for trial in trials], dtype=np.int64)
        rates, _ = _pool_trial_rate_matrix(nwb_path, trials)
        if signal_view == "pseudo_mua":
            with NWBHDF5IO(str(nwb_path), "r") as io:
                nwb = io.read()
                if nwb.units is None:
                    raise ValueError(f"NWB file has no units table: {nwb_path}")
                electrode_ids = electrode_ids_from_units(nwb.units.to_dataframe())
            rates, _ = pool_trial_rates_by_electrode(rates, electrode_ids)
        coefficients, variance, design, _rank, _condition = _trial_level_t4_ols(rates, directions)
        beta_chunks.append(coefficients[:, 1:3])
        noise_trace_chunks.append(variance * float(np.trace(np.linalg.inv(design.T @ design)[1:3, 1:3])))
    beta = np.concatenate(beta_chunks, axis=0)
    noise_trace = np.concatenate(noise_trace_chunks, axis=0)
    raw_second_moment = float(np.mean(np.sum(np.square(beta), axis=1)) / 2.0)
    expected_noise_variance = float(np.mean(noise_trace) / 2.0)
    receipt: dict[str, object] = {
        "formula_version": T4R_POSTERIOR_FORMULA_VERSION,
        "prior_variance": max(T4R_PRIOR_VARIANCE_FLOOR, raw_second_moment - expected_noise_variance),
        "pool_size": pool_size,
        "source_sessions": [session_name_from_path(path) for path in train_files],
        "source_fingerprints": [_source_fingerprint(path) for path in train_files],
        "source_unit_count": int(beta.shape[0]),
        "raw_direction_second_moment": raw_second_moment,
        "expected_ols_noise_variance": expected_noise_variance,
        "target_sessions_used": False,
    }
    receipt["prior_sha256"] = _t4r_prior_hash(receipt)
    return receipt


def posterior_mean_t4_with_covariance(
    trial_rates: np.ndarray,
    direction_indices: np.ndarray,
    *,
    prior_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Posterior mean and the 2D directional covariance, with flat intercept prior."""
    if not math.isfinite(prior_variance) or prior_variance < T4R_PRIOR_VARIANCE_FLOOR:
        raise ValueError("t4r prior_variance must be finite and positive")
    coefficients, variance, design, rank, condition = _trial_level_t4_ols(trial_rates, direction_indices)
    directions = np.asarray(direction_indices, dtype=np.int64)
    response = np.asarray(trial_rates, dtype=np.float64)[:, directions >= 0]
    xtx = design.T @ design
    xty = design.T @ response.T
    posterior = np.empty_like(coefficients)
    covariance_ac = np.empty((coefficients.shape[0], 2, 2), dtype=np.float64)
    for index, sigma2 in enumerate(variance):
        system = xtx + np.diag([0.0, float(sigma2) / prior_variance, float(sigma2) / prior_variance])
        posterior[index] = np.linalg.solve(system, xty[:, index])
        covariance_ac[index] = float(sigma2) * np.linalg.inv(system)[1:3, 1:3]
    features = np.column_stack((
        posterior[:, 1], posterior[:, 2],
        np.hypot(posterior[:, 1], posterior[:, 2]), posterior[:, 0],
    ))
    if not np.isfinite(features).all() or not np.isfinite(covariance_ac).all():
        raise ValueError("t4r posterior mean/covariance is non-finite")
    return features.astype(np.float32), variance.astype(np.float32), covariance_ac.astype(np.float32), rank, condition


def posterior_mean_t4(trial_rates: np.ndarray, direction_indices: np.ndarray, *, prior_variance: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Closed-form posterior mean with an unpenalized intercept."""
    features, variance, _covariance_ac, rank, condition = posterior_mean_t4_with_covariance(
        trial_rates, direction_indices, prior_variance=prior_variance,
    )
    return features, variance, rank, condition


def posterior_angular_reliability(
    posterior_t4: np.ndarray,
    covariance_ac: np.ndarray,
    *,
    eps: float = T4RQ_ANGULAR_EPS,
    zero_modulation_reliability: float = T4RQ_ZERO_MODULATION_RELIABILITY,
) -> np.ndarray:
    """SO(2)-invariant negative log posterior angular variance for E03."""
    features = np.asarray(posterior_t4, dtype=np.float64)
    covariance = np.asarray(covariance_ac, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError(f"posterior_t4 must be [units,4], got {features.shape}")
    if covariance.shape != (features.shape[0], 2, 2):
        raise ValueError("covariance_ac must be [units,2,2] aligned to posterior_t4")
    if not np.isfinite(features).all() or not np.isfinite(covariance).all():
        raise ValueError("posterior angular reliability inputs must be finite")
    mu = features[:, :2]
    modulation = np.hypot(mu[:, 0], mu[:, 1])
    u_perp = np.column_stack((-mu[:, 1], mu[:, 0])) / (modulation[:, None] + eps)
    angular_variance = np.einsum("ni,nij,nj->n", u_perp, covariance, u_perp) / (np.square(modulation) + eps)
    reliability = -np.log(np.maximum(angular_variance, 0.0) + eps)
    reliability[modulation <= MODULATION_EPS] = zero_modulation_reliability
    if not np.isfinite(reliability).all():
        raise ValueError("posterior angular reliability is non-finite")
    return reliability.astype(np.float32)

def _unit_tuning_features(
    trial_rates: np.ndarray,
    direction_indices: np.ndarray,
    present_directions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """T4/T8 features for one unit, plus ``(is_zero_spike, is_zero_modulation)`` flags.

    ``trial_rates`` is this unit's per-pool-trial firing rate (Hz), ``direction_indices`` is
    the parallel per-pool-trial canonical direction index (``-1`` for a trial with no usable
    ``target_dir``), and ``present_directions`` is the sorted set of distinct direction indices
    with at least one pool trial. The caller guarantees ``len(present_directions) >= 2``: the
    "fewer than 2 distinct directions" degeneracy is a session-wide fact (every unit shares the
    same pool trials/directions) handled once by the orchestrator below, not re-detected here.

    A unit with zero spikes across every pool trial gets the exact all-zero fill for both T4 and
    T8: for T8 that zero is the *correct* rate (no observed spikes -> 0 Hz in every direction),
    and for T4 it is the limit of the fit for an identically-zero target vector (b=m=0), not an
    arbitrary sentinel -- so this never disagrees with what an unguarded fit would have produced,
    it just avoids relying on that implicitly and gives an explicit, metadata-counted reason.
    """
    t8 = np.zeros(TUNING_NUM_DIRECTIONS, dtype=np.float32)
    if not np.any(trial_rates):
        return np.zeros(4, dtype=np.float32), t8, True, False

    thetas = np.empty(len(present_directions), dtype=np.float64)
    per_direction_mean = np.empty(len(present_directions), dtype=np.float64)
    for row, direction_index in enumerate(present_directions):
        mask = direction_indices == direction_index
        mean_rate = float(trial_rates[mask].mean())
        t8[direction_index] = mean_rate
        per_direction_mean[row] = mean_rate
        thetas[row] = CANONICAL_DIRECTIONS_RAD[direction_index]

    a, c, m, b = _fit_cosine_tuning(thetas, per_direction_mean)
    # See SIDE_FEATURE_DIMS/E3_E4_ENCODER_PROGRAM.md section 1.4 (also restated on
    # compute_unit_side_features_uncached below): we emit [m*cos(phi), m*sin(phi), m, b] = [a,
    # c, m, b], never [cos(phi), sin(phi), m, b]. a and c are already rate-scaled (same units
    # and magnitude as m and b), so per-column train-only z-scoring (fit_side_feature_stats /
    # _fit_robust_stats, applied uniformly to all SIDE_FEATURE_DIMS columns downstream) treats
    # them exactly like any other rate-like scalar. Emitting bare cos(phi)/sin(phi) instead
    # would be a pure direction unit vector on [-1, 1] with no rate information, and z-scoring
    # its two components independently (different train std per column) would rescale it into
    # an ellipse and destroy the "this pair is a single angle" geometry -- so that
    # decomposition is deliberately avoided rather than normalized around.
    t4 = np.array([a, c, m, b], dtype=np.float32)
    is_zero_modulation = m <= MODULATION_EPS
    return t4, t8, False, is_zero_modulation


def _pool_trial_rate_matrix(nwb_path: Path, pool_trials: Sequence[dict]) -> tuple[np.ndarray, int]:
    """Per-unit, per-pool-trial firing rate (Hz), shape ``[num_units, len(pool_trials)]``.

    Counts only spikes within ``[start_time, stop_time)`` of each pool trial (the same
    half-open convention ``np.searchsorted`` default ``side="left"`` gives the bin-edge
    filtering elsewhere in this file/``multisession_datamodule.py``) -- never spikes from
    outside the pool trial windows (leakage discipline, E3_E4_ENCODER_PROGRAM.md section 3).
    """
    starts = np.asarray([trial["start_time"] for trial in pool_trials], dtype=np.float64)
    stops = np.asarray([trial["stop_time"] for trial in pool_trials], dtype=np.float64)
    durations = stops - starts
    if np.any(durations <= 0):
        raise ValueError("Pool trial stop_time must be strictly after start_time")

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        units_df = nwb.units.to_dataframe()
        num_units = len(units_df)
        rates = np.zeros((num_units, len(pool_trials)), dtype=np.float64)
        for unit_idx in range(num_units):
            spike_times = np.asarray(units_df.iloc[unit_idx]["spike_times"], dtype=np.float64)
            if spike_times.size and not np.all(spike_times[:-1] <= spike_times[1:]):
                raise ValueError(f"Unit {unit_idx}: spike_times must be non-decreasing")
            start_idx = np.searchsorted(spike_times, starts)
            stop_idx = np.searchsorted(spike_times, stops)
            rates[unit_idx] = (stop_idx - start_idx).astype(np.float64) / durations
    return rates, num_units


def _compute_tuning_features_uncached(
    nwb_path: Path,
    *,
    feature_group: str,
    pool_size: int,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
    signal_view: str = "sua",
    posterior_prior: dict[str, object] | None = None,
) -> tuple[np.ndarray, SideFeatureMetadata]:
    """E3 directional tuning features (T4/T8), computed from the first ``pool_size`` rewarded
    trials only -- the identical pool boundary F1/F2 use, via the same
    ``list_datamodule_rewarded_trials`` filter (not reimplemented)."""
    if feature_group not in TUNING_FEATURE_NAMES:
        raise ValueError(f"Unsupported tuning feature_group {feature_group!r}")
    _validate_signal_view(signal_view)

    pool_trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
    )
    if len(pool_trials) < pool_size:
        raise ValueError(
            f"{session_name_from_path(nwb_path)}: only {len(pool_trials)} rewarded trials "
            f"pass the datamodule filter; pool_size={pool_size} required"
        )
    pool_trials = pool_trials[:pool_size]

    direction_indices = np.array(
        [
            _nearest_canonical_direction_index(trial["target_dir"])
            if trial.get("target_dir") is not None
            else -1
            for trial in pool_trials
        ],
        dtype=np.int64,
    )
    present_directions = sorted({int(index) for index in direction_indices if index >= 0})

    rates, num_units = _pool_trial_rate_matrix(nwb_path, pool_trials)
    # In pseudo-MUA view, calibration trial rates are aggregated by physical
    # electrode *before* the per-direction means and cosine fit are computed.
    # Never average sorted-unit T4 values: that would in general differ from a
    # fit to the summed channel spike train.
    if signal_view == "pseudo_mua":
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            if nwb.units is None:
                raise ValueError(f"NWB file has no units table: {nwb_path}")
            electrode_ids = electrode_ids_from_units(nwb.units.to_dataframe())
        rates, _ = pool_trial_rates_by_electrode(rates, electrode_ids)
    num_channels = rates.shape[0]

    t4 = np.zeros((num_channels, 4), dtype=np.float32)
    t8 = np.zeros((num_channels, TUNING_NUM_DIRECTIONS), dtype=np.float32)
    confidence = np.zeros((num_channels, 2), dtype=np.float32)
    zero_spike = 0
    zero_modulation = 0
    insufficient_direction = 0

    design_rank = 0
    design_condition = math.inf
    if present_directions:
        theta = np.asarray([CANONICAL_DIRECTIONS_RAD[index] for index in present_directions])
        design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
        design_rank = int(np.linalg.matrix_rank(design))
        design_condition = float(np.linalg.cond(design)) if design_rank == 3 else math.inf
    if feature_group in {"t4c", "t4w3", "t4r", "t4rq", "t4rql"} and (
        design_rank != 3 or not math.isfinite(design_condition)
    ):
        raise ValueError(
            f"{session_name_from_path(nwb_path)}: {feature_group} requires "
            "rank=3 finite-condition design; "
            f"got rank={design_rank}, condition={design_condition}"
        )

    if feature_group in {"t4r", "t4rq", "t4rql"} and posterior_prior is None:
        raise ValueError("t4r features require a source-only posterior_prior receipt")

    if len(present_directions) < 2:
        # Session-wide degeneracy shared by every unit (E3_E4_ENCODER_PROGRAM.md section 1.4):
        # with fewer than 2 distinct pool directions, b/a/c are not identifiable from any
        # unit's data (an underdetermined linear system with <2 independent equations), so
        # every unit gets the fixed all-zero fill rather than an arbitrary lstsq minimum-norm
        # answer. Expected to essentially never trigger in practice (ROADMAP.md: the first 30
        # rewarded trials already cover all 8 directions on the sessions checked), but must
        # still degrade to a fixed, counted, finite value rather than raising or emitting NaN.
        insufficient_direction = num_channels
    else:
        for unit_idx in range(num_channels):
            unit_t4, unit_t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
                rates[unit_idx], direction_indices, present_directions
            )
            t4[unit_idx] = unit_t4
            t8[unit_idx] = unit_t8
            if feature_group in {"t4c", "t4w3"}:
                confidence[unit_idx] = tuning_fit_confidence_descriptor(
                    rates[unit_idx], direction_indices, selected_t4=unit_t4
                )
            zero_spike += int(is_zero_spike)
            zero_modulation += int(is_zero_modulation)

    semantic_version = feature_semantics_version(feature_group)
    cache_payload = {
        "feature_version": semantic_version,
        "feature_group": feature_group,
        "pool_size": pool_size,
        **_pool_context_key(
            bin_size_ms=bin_size_ms, window_size=window_size, trial_result_filter=trial_result_filter
        ),
        "source": _source_fingerprint(nwb_path),
    }
    if signal_view == "pseudo_mua":
        cache_payload["signal_view"] = signal_view
        cache_payload["electrode_mapping"] = _electrode_mapping_fingerprint(nwb_path)
    metadata = SideFeatureMetadata(
        feature_group=feature_group,
        feature_version=semantic_version,
        pool_size=pool_size,
        cache_key=_cache_key(cache_payload),
        degenerate_unit_count=zero_spike + zero_modulation + insufficient_direction,
        zero_spike_unit_count=zero_spike,
        single_spike_unit_count=0,
        zero_noise_std_unit_count=0,
        zero_template_max_unit_count=0,
        zero_modulation_unit_count=zero_modulation,
        insufficient_direction_unit_count=insufficient_direction,
    )
    if feature_group == "t4w3":
        features, _shrink_factors = uncertainty_wiener_shrink_t4(
            t4,
            confidence[:, 0],
            direction_indices,
        )
    elif feature_group in {"t4r", "t4rq", "t4rql"}:
        assert posterior_prior is not None
        posterior_t4, _variance, covariance_ac, design_rank, design_condition = posterior_mean_t4_with_covariance(
            rates, direction_indices,
            prior_variance=float(posterior_prior["prior_variance"]),
        )
        features = posterior_t4
        if feature_group in {"t4rq", "t4rql"}:
            reliability = posterior_angular_reliability(posterior_t4, covariance_ac)
            features = np.concatenate((posterior_t4, reliability[:, None]), axis=1)
        posterior_cache_key = _cache_key({
            "feature_version": feature_semantics_version(feature_group),
            "feature_group": feature_group,
            "pool_size": pool_size,
            **_pool_context_key(bin_size_ms=bin_size_ms, window_size=window_size, trial_result_filter=trial_result_filter),
            "source": _source_fingerprint(nwb_path),
            "posterior_prior_sha256": str(posterior_prior["prior_sha256"]),
        })
        metadata = SideFeatureMetadata(
            **{**metadata.__dict__,
               "cache_key": posterior_cache_key,
               "posterior_prior_variance": float(posterior_prior["prior_variance"]),
               "posterior_prior_sha256": str(posterior_prior["prior_sha256"]),
               "posterior_design_rank": design_rank,
               "posterior_design_condition": design_condition,
               "posterior_reliability_formula": "angular_posterior_variance_q3_v1" if feature_group in {"t4rq", "t4rql"} else "",
               "posterior_reliability_epsilon": T4RQ_ANGULAR_EPS if feature_group in {"t4rq", "t4rql"} else 0.0,
               "posterior_reliability_zero_floor": T4RQ_ZERO_MODULATION_RELIABILITY if feature_group in {"t4rq", "t4rql"} else 0.0}
        )
    elif feature_group == "t4":
        features = t4
    elif feature_group == "t4c":
        features = np.concatenate([t4, confidence], axis=1)
    else:
        features = t8
    return features, metadata



def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array).astype(np.float32, copy=False).copy(order="C")
    return hashlib.sha256(value.tobytes()).hexdigest()


def _trial_alignment_time(trial: dict[str, float]) -> tuple[float, str]:
    for key in ("go_cue_time", "target_on_time", "start_time"):
        value = trial.get(key)
        if value is not None and np.isfinite(value):
            return float(value), key
    raise ValueError("Template-Ridge trial has no finite alignment event")


def _cursor_velocity_interpolator(nwb_path: Path):
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        vel_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        data = np.asarray(vel_series.data[:], dtype=np.float64)
        if vel_series.timestamps is not None:
            timestamps = np.asarray(vel_series.timestamps[:], dtype=np.float64)
        else:
            timestamps = (
                float(vel_series.starting_time)
                + np.arange(data.shape[0], dtype=np.float64) / float(vel_series.rate)
            )
    return timestamps, data


def learn_template_ridge_speed_profile(
    train_files: Sequence[Path],
    *,
    pool_size: int,
    bin_size_ms: int = 20,
    window_size: int = 50,
    trial_result_filter: str = "R",
) -> dict[str, object]:
    """Learn a source-only go-cue aligned scalar speed profile for Template-Ridge."""
    dt = bin_size_ms / 1000.0
    samples: list[np.ndarray] = []
    alignment_events: dict[str, int] = {}
    for nwb_path in train_files:
        trials = list_datamodule_rewarded_trials(
            nwb_path,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        )
        timestamps, velocity = _cursor_velocity_interpolator(nwb_path)
        for trial in trials[:pool_size]:
            if trial.get("target_dir") is None:
                continue
            align, event_name = _trial_alignment_time(trial)
            alignment_events[event_name] = alignment_events.get(event_name, 0) + 1
            grid = align + np.arange(window_size, dtype=np.float64) * dt
            if grid[-1] > float(trial["stop_time"]):
                continue
            vx = np.interp(grid, timestamps, velocity[:, 0])
            vy = np.interp(grid, timestamps, velocity[:, 1])
            speed = np.hypot(vx, vy)
            if np.isfinite(speed).all():
                samples.append(speed.astype(np.float64))
    if not samples:
        raise ValueError("Template-Ridge speed profile found no source trials")
    raw = np.stack(samples, axis=0).mean(axis=0)
    peak = float(np.max(raw))
    if peak <= 1.0e-8 or not math.isfinite(peak):
        raise ValueError("Template-Ridge speed profile has zero/nonfinite peak")
    profile = (raw / peak).astype(np.float32)
    return {
        "profile": profile,
        "profile_sha256": _array_sha256(profile),
        "source_trial_count": len(samples),
        "raw_peak_speed": peak,
        "alignment_event": max(alignment_events, key=alignment_events.get),
        "alignment_event_counts": alignment_events,
        "source_sessions": [session_name_from_path(path) for path in train_files],
    }


def _binned_spike_matrix(nwb_path: Path, *, bin_size_ms: int) -> tuple[np.ndarray, np.ndarray]:
    bin_size_s = bin_size_ms / 1000.0
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        units_df = nwb.units.to_dataframe()
        all_spikes = np.concatenate(units_df["spike_times"].values)
        t_min = float(all_spikes.min())
        t_max = float(all_spikes.max())
        bin_edges = np.arange(t_min, t_max + bin_size_s, bin_size_s)
        counts = np.zeros((len(bin_edges) - 1, len(units_df)), dtype=np.float32)
        for unit_idx, spike_times in enumerate(units_df["spike_times"].values):
            counts[:, unit_idx] = np.histogram(np.asarray(spike_times, dtype=np.float64), bins=bin_edges)[0]
    return counts, bin_edges


def _weighted_dual_ridge(design: np.ndarray, target: np.ndarray, weights: np.ndarray, ridge: float) -> tuple[np.ndarray, float, float]:
    if design.ndim != 2 or target.ndim != 2 or target.shape[0] != design.shape[0]:
        raise ValueError("Template-Ridge design/target shape mismatch")
    if weights.shape != (design.shape[0],):
        raise ValueError("Template-Ridge weights shape mismatch")
    keep = weights > 1.0e-8
    x = design[keep].astype(np.float64, copy=False)
    y = target[keep].astype(np.float64, copy=False)
    w = weights[keep].astype(np.float64, copy=False)
    if x.shape[0] < 4:
        raise ValueError("Template-Ridge needs at least four weighted rows")
    sw = np.sqrt(w)[:, None]
    xw = x * sw
    yw = y * sw
    coefficients = []
    conditions = []
    damp = math.sqrt(float(ridge))
    for output_idx in range(yw.shape[1]):
        result = lsqr(
            xw,
            yw[:, output_idx],
            damp=damp,
            atol=1.0e-5,
            btol=1.0e-5,
            iter_lim=1000,
        )
        coefficients.append(result[0])
        conditions.append(float(result[6]))
    beta = np.stack(coefficients, axis=1)
    condition = max(conditions) if conditions else math.inf
    # Exact hat-matrix trace would require a dense solve; for the experiment
    # receipt this stable rank cap records the identifiable-row scale without
    # making the audit as expensive as training.
    trace_hat = float(min(xw.shape))
    return beta.astype(np.float32), condition, trace_hat


def _compute_template_ridge_features_uncached(
    nwb_path: Path,
    *,
    feature_group: str,
    pool_size: int,
    template_profile: np.ndarray,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str,
    signal_view: str = "sua",
) -> tuple[np.ndarray, SideFeatureMetadata]:
    if feature_group not in TEMPLATE_RIDGE_FEATURE_NAMES:
        raise ValueError(f"Unsupported Template-Ridge feature_group {feature_group!r}")
    _validate_signal_view(signal_view)
    if signal_view != "sua":
        raise ValueError("Template-Ridge D-b is currently implemented for sorted SUA only")
    profile = np.asarray(template_profile, dtype=np.float32).reshape(-1)
    if profile.shape != (window_size,) or not np.isfinite(profile).all():
        raise ValueError(f"Template-Ridge profile must be finite [{window_size}], got {profile.shape}")

    trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
    )
    if len(trials) < pool_size:
        raise ValueError(
            f"{session_name_from_path(nwb_path)}: only {len(trials)} rewarded trials; pool_size={pool_size} required"
        )
    pool_trials = trials[:pool_size]
    if feature_group == "trls4":
        valid_indices = [i for i, trial in enumerate(pool_trials) if trial.get("target_dir") is not None]
        permutation = np.asarray(valid_indices, dtype=np.int64)
        digest = hashlib.sha256(
            f"template-ridge-label-shuffle-v1:{TEMPLATE_RIDGE_LABEL_SHUFFLE_SEED}:{session_name_from_path(nwb_path)}".encode()
        ).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
        shuffled = permutation[rng.permutation(permutation.size)]
        if np.array_equal(shuffled, permutation) and shuffled.size > 1:
            shuffled = np.roll(shuffled, 1)
        shuffled_dirs = {int(src): pool_trials[int(dst)]["target_dir"] for src, dst in zip(permutation, shuffled)}
    else:
        shuffled_dirs = {}

    binned, bin_edges = _binned_spike_matrix(nwb_path, bin_size_ms=bin_size_ms)
    num_units = binned.shape[1]
    dt = bin_size_ms / 1000.0
    design_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weights: list[float] = []
    support_counts = np.zeros(num_units, dtype=np.float64)
    support_bins = 0
    alignment_event = ""
    for trial_index, trial in enumerate(pool_trials):
        direction = shuffled_dirs.get(trial_index, trial.get("target_dir"))
        if direction is None or not np.isfinite(direction):
            continue
        align, event_name = _trial_alignment_time(trial)
        alignment_event = alignment_event or event_name
        direction_vec = np.asarray([math.cos(float(direction)), math.sin(float(direction))], dtype=np.float64)
        for offset, speed in enumerate(profile):
            time = align + offset * dt
            if time >= float(trial["stop_time"]):
                continue
            end_bin = int(np.searchsorted(bin_edges, time, side="right"))
            start_bin = end_bin - window_size
            window = np.zeros((window_size, num_units), dtype=np.float32)
            src_start = max(0, start_bin)
            src_stop = min(end_bin, binned.shape[0])
            if src_stop > src_start:
                dst_start = window_size - (src_stop - src_start)
                window[dst_start:] = binned[src_start:src_stop]
            design_rows.append(window.reshape(-1))
            target_rows.append(float(speed) * direction_vec)
            weights.append(float(speed) ** 2)
        start_bin = int(trial["start"])
        stop_bin = int(trial["stop"])
        if stop_bin > start_bin:
            support_counts += binned[start_bin:stop_bin].sum(axis=0)
            support_bins += stop_bin - start_bin
    if not design_rows or support_bins <= 0:
        raise ValueError(f"{session_name_from_path(nwb_path)}: Template-Ridge produced no rows")
    design = np.stack(design_rows, axis=0)
    target = np.stack(target_rows, axis=0)
    weight_array = np.asarray(weights, dtype=np.float64)
    beta, condition, trace_hat = _weighted_dual_ridge(design, target, weight_array, TEMPLATE_RIDGE_RIDGE)
    beta = beta.reshape(window_size, num_units, 2)
    projected = np.tensordot(profile.astype(np.float32), beta, axes=([0], [0]))
    support_rate = support_counts / max(support_bins * dt, 1.0e-8)
    features = np.stack(
        [projected[:, 0], projected[:, 1], np.linalg.norm(projected, axis=1), support_rate],
        axis=1,
    ).astype(np.float32)
    if features.shape != (num_units, 4) or not np.isfinite(features).all():
        raise ValueError(f"Invalid Template-Ridge features for {nwb_path}: {features.shape}")
    cache_payload = {
        "feature_version": feature_semantics_version(feature_group),
        "feature_group": feature_group,
        "pool_size": pool_size,
        "ridge": TEMPLATE_RIDGE_RIDGE,
        "template_profile_sha256": _array_sha256(profile),
        **_pool_context_key(
            bin_size_ms=bin_size_ms, window_size=window_size, trial_result_filter=trial_result_filter
        ),
        "source": _source_fingerprint(nwb_path),
    }
    zero_spike = int(np.sum(support_counts == 0))
    metadata = SideFeatureMetadata(
        feature_group=feature_group,
        feature_version=feature_semantics_version(feature_group),
        pool_size=pool_size,
        cache_key=_cache_key(cache_payload),
        degenerate_unit_count=zero_spike,
        zero_spike_unit_count=zero_spike,
        single_spike_unit_count=0,
        zero_noise_std_unit_count=0,
        zero_template_max_unit_count=0,
        template_ridge_constructed_rows=int(design.shape[0]),
        template_ridge_feature_count=int(design.shape[1]),
        template_ridge_condition=condition,
        template_ridge_trace_hat=trace_hat,
        template_ridge_profile_sha256=_array_sha256(profile),
        template_ridge_alignment_event=alignment_event,
    )
    return features, metadata

def compute_unit_side_features_uncached(
    nwb_path: Path,
    *,
    feature_group: str,
    pool_size: int,
    bin_size_ms: int = 20,
    window_size: int = 50,
    trial_result_filter: str = "R",
    pool_end_time: float | None = None,
    signal_view: str = "sua",
    template_profile: np.ndarray | None = None,
    posterior_prior: dict[str, object] | None = None,
) -> tuple[np.ndarray, SideFeatureMetadata]:
    _validate_signal_view(signal_view)
    if feature_group in TEMPLATE_RIDGE_FEATURE_NAMES:
        if template_profile is None:
            raise ValueError("Template-Ridge features require a source-only template_profile")
        return _compute_template_ridge_features_uncached(
            nwb_path,
            feature_group=feature_group,
            pool_size=pool_size,
            template_profile=template_profile,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
        )
    if feature_group in TUNING_FEATURE_NAMES:
        # T4/T8 are computed per pool *trial* (direction-conditioned mean firing rate), not
        # from a single spike-time prefix cutoff, so pool_end_time -- only meaningful for the
        # waveform prefix read below -- does not apply here. The leakage boundary is still
        # exactly the same pool_size'th rewarded trial (_compute_tuning_features_uncached
        # slices list_datamodule_rewarded_trials(...)[:pool_size], the identical trial list
        # calibration_pool_end_time itself is built from).
        return _compute_tuning_features_uncached(
            nwb_path,
            feature_group=feature_group,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
            posterior_prior=posterior_prior,
        )
    if signal_view != "sua":
        raise ValueError(
            "pseudo_mua side features currently support only directional tuning "
            "groups ('t4'/'t8'); waveform features are sorted-unit only"
        )
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unsupported feature_group {feature_group!r}")
    feature_names = FEATURE_GROUPS[feature_group]

    if pool_end_time is None:
        pool_end_time = calibration_pool_end_time(
            nwb_path,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        )

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        units_df = nwb.units.to_dataframe()
        num_units = len(units_df)
        if "waveforms" not in nwb.units.colnames:
            raise ValueError(f"NWB units table has no waveforms column: {nwb_path}")

    features = np.zeros((num_units, len(feature_names)), dtype=np.float32)
    zero_spike = single_spike = zero_noise = zero_template_max = 0

    with h5py.File(nwb_path, "r") as handle:
        waveforms = handle["units/waveforms"]
        waveforms_index = handle["units/waveforms_index"]
        waveforms_index_index = np.asarray(handle["units/waveforms_index_index"])

        for unit_idx in range(num_units):
            spike_start, spike_end = _unit_spike_bounds(waveforms_index_index, unit_idx)
            spike_times = np.asarray(units_df.iloc[unit_idx]["spike_times"], dtype=np.float64)
            expected_spikes = spike_end - spike_start
            if spike_times.size != expected_spikes:
                raise ValueError(
                    f"Unit {unit_idx}: spike_times length {spike_times.size} != "
                    f"waveforms_index span {expected_spikes}"
                )

            num_in_pool, _ = _in_pool_spike_prefix(spike_times, pool_end_time)
            if num_in_pool == 0:
                zero_spike += 1
                continue
            if num_in_pool == 1:
                single_spike += 1

            stacked = _read_unit_waveform_block(
                waveforms, waveforms_index, spike_start, num_in_pool
            )
            template = stacked.mean(axis=0)
            residuals = stacked - template
            scalars = _scalar_features_from_template(template, residuals.reshape(-1))
            if scalars["noise_std"] <= NOISE_STD_EPS:
                zero_noise += 1
            if abs(float(template.max())) <= TEMPLATE_MAX_EPS:
                zero_template_max += 1
            for col_idx, name in enumerate(feature_names):
                value = scalars[name]
                if not np.isfinite(value):
                    value = 0.0
                features[unit_idx, col_idx] = value

    degenerate = zero_spike + single_spike
    semantic_version = feature_semantics_version(feature_group)
    cache_payload = {
        "feature_version": semantic_version,
        "feature_group": feature_group,
        "pool_size": pool_size,
        **_pool_context_key(
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        ),
        "source": _source_fingerprint(nwb_path),
    }
    if feature_group in {"t4r", "t4rq", "t4rql"}:
        assert posterior_prior is not None
        cache_payload["posterior_prior_sha256"] = str(posterior_prior["prior_sha256"])
    metadata = SideFeatureMetadata(
        feature_group=feature_group,
        feature_version=semantic_version,
        pool_size=pool_size,
        cache_key=_cache_key(cache_payload),
        degenerate_unit_count=degenerate,
        zero_spike_unit_count=zero_spike,
        single_spike_unit_count=single_spike,
        zero_noise_std_unit_count=zero_noise,
        zero_template_max_unit_count=zero_template_max,
    )
    return features, metadata


def _fit_robust_stats(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    q_low, q_high = TRAIN_CLIP_QUANTILES
    lower = np.quantile(stacked, q_low, axis=0)
    upper = np.quantile(stacked, q_high, axis=0)
    clipped = np.clip(stacked, lower, upper)
    clipped_count = int(np.sum((stacked < lower) | (stacked > upper)))
    mean = clipped.mean(axis=0).astype(np.float32)
    std = clipped.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std, clipped_count


def fit_side_feature_stats(
    train_files: Sequence[Path],
    *,
    feature_group: str,
    pool_size: int,
    cache_dir: Path | None = None,
    bin_size_ms: int = 20,
    window_size: int = 50,
    trial_result_filter: str = "R",
    signal_view: str = "sua",
    return_template_receipt: bool = False,
    return_t4r_receipt: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Train-only mean/std for z-scoring side features."""
    if feature_group not in KNOWN_FEATURE_GROUPS:
        raise ValueError(f"Unsupported feature_group {feature_group!r}")
    _validate_signal_view(signal_view)

    template_receipt: dict[str, object] | None = None
    template_profile: np.ndarray | None = None
    template_profile_hash: str | None = None
    posterior_prior: dict[str, object] | None = None
    posterior_prior_hash: str | None = None
    if feature_group in {"t4r", "t4rq", "t4rql"}:
        posterior_prior = fit_t4r_posterior_prior(
            train_files, pool_size=pool_size, bin_size_ms=bin_size_ms,
            window_size=window_size, trial_result_filter=trial_result_filter,
            signal_view=signal_view,
        )
        posterior_prior_hash = str(posterior_prior["prior_sha256"])
    if feature_group in TEMPLATE_RIDGE_FEATURE_NAMES:
        template_receipt = learn_template_ridge_speed_profile(
            train_files,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
        )
        template_profile = np.asarray(template_receipt["profile"], dtype=np.float32)
        template_profile_hash = str(template_receipt["profile_sha256"])

    if cache_dir is not None:
        cache_path = _side_stats_cache_path(
            cache_dir,
            train_files,
            feature_group=feature_group,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
            template_profile_hash=template_profile_hash,
            posterior_prior_hash=posterior_prior_hash,
        )
        with _exclusive_cache_lock(cache_path):
            if cache_path.is_file():
                try:
                    with np.load(cache_path, allow_pickle=False) as cache:
                        mean = cache["mean"].astype(np.float32, copy=False)
                        std = cache["std"].astype(np.float32, copy=False)
                    logger.info("Loaded cached side-feature statistics from %s", cache_path)
                    if return_template_receipt:
                        assert template_receipt is not None
                        return mean, std, template_receipt
                    if return_t4r_receipt:
                        assert posterior_prior is not None
                        return mean, std, posterior_prior
                    return mean, std
                except (KeyError, OSError, ValueError) as exc:
                    logger.warning("Discarding unreadable side-feature stats cache %s: %s", cache_path, exc)
                    cache_path.unlink(missing_ok=True)

    chunks: list[np.ndarray] = []
    for nwb_path in train_files:
        raw, _ = compute_unit_side_features_uncached(
            nwb_path,
            feature_group=feature_group,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
            template_profile=template_profile,
            posterior_prior=posterior_prior,
        )
        chunks.append(raw)
    stacked = np.concatenate(chunks, axis=0)
    mean, std, clipped_count = _fit_robust_stats(stacked)
    logger.info(
        "Fitted side-feature stats for %s: clipped %d scalar values before mean/std",
        feature_group,
        clipped_count,
    )

    if cache_dir is not None:
        cache_path = _side_stats_cache_path(
            cache_dir,
            train_files,
            feature_group=feature_group,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
            template_profile_hash=template_profile_hash,
            posterior_prior_hash=posterior_prior_hash,
        )
        with _exclusive_cache_lock(cache_path):
            arrays = {
                "mean": mean,
                "std": std,
                "clipped_count": np.asarray(clipped_count, dtype=np.int64),
            }
            if template_profile is not None and template_receipt is not None:
                arrays["template_profile"] = template_profile
                arrays["template_profile_sha256"] = np.asarray(template_receipt["profile_sha256"])
            _write_npz_atomically(cache_path, **arrays)
            logger.info("Cached side-feature statistics at %s", cache_path)
    if return_template_receipt:
        return mean, std, template_receipt or {}
    if return_t4r_receipt:
        return mean, std, posterior_prior or {}
    return mean, std

def _metadata_to_cache_arrays(metadata: SideFeatureMetadata) -> dict[str, np.ndarray]:
    return {
        "feature_group": np.asarray(metadata.feature_group),
        "feature_version": np.asarray(metadata.feature_version),
        "pool_size": np.asarray(metadata.pool_size),
        "cache_key": np.asarray(metadata.cache_key),
        "degenerate_unit_count": np.asarray(metadata.degenerate_unit_count),
        "zero_spike_unit_count": np.asarray(metadata.zero_spike_unit_count),
        "single_spike_unit_count": np.asarray(metadata.single_spike_unit_count),
        "zero_noise_std_unit_count": np.asarray(metadata.zero_noise_std_unit_count),
        "zero_template_max_unit_count": np.asarray(metadata.zero_template_max_unit_count),
        "zero_modulation_unit_count": np.asarray(metadata.zero_modulation_unit_count),
        "insufficient_direction_unit_count": np.asarray(metadata.insufficient_direction_unit_count),
        "template_ridge_constructed_rows": np.asarray(metadata.template_ridge_constructed_rows),
        "template_ridge_feature_count": np.asarray(metadata.template_ridge_feature_count),
        "template_ridge_condition": np.asarray(metadata.template_ridge_condition),
        "template_ridge_trace_hat": np.asarray(metadata.template_ridge_trace_hat),
        "template_ridge_profile_sha256": np.asarray(metadata.template_ridge_profile_sha256),
        "template_ridge_alignment_event": np.asarray(metadata.template_ridge_alignment_event),
        "posterior_prior_variance": np.asarray(metadata.posterior_prior_variance),
        "posterior_prior_sha256": np.asarray(metadata.posterior_prior_sha256),
        "posterior_design_rank": np.asarray(metadata.posterior_design_rank),
        "posterior_design_condition": np.asarray(metadata.posterior_design_condition),
        "posterior_reliability_formula": np.asarray(metadata.posterior_reliability_formula),
        "posterior_reliability_epsilon": np.asarray(metadata.posterior_reliability_epsilon),
        "posterior_reliability_zero_floor": np.asarray(metadata.posterior_reliability_zero_floor),
    }


def _metadata_from_cache(cache) -> SideFeatureMetadata:
    return SideFeatureMetadata(
        feature_group=str(cache["feature_group"].item()),
        feature_version=int(cache["feature_version"].item()),
        pool_size=int(cache["pool_size"].item()),
        cache_key=str(cache["cache_key"].item()),
        degenerate_unit_count=int(cache["degenerate_unit_count"].item()),
        zero_spike_unit_count=int(cache["zero_spike_unit_count"].item()),
        single_spike_unit_count=int(cache["single_spike_unit_count"].item()),
        zero_noise_std_unit_count=int(cache["zero_noise_std_unit_count"].item()),
        zero_template_max_unit_count=int(cache["zero_template_max_unit_count"].item()),
        zero_modulation_unit_count=int(cache["zero_modulation_unit_count"].item()),
        insufficient_direction_unit_count=int(cache["insufficient_direction_unit_count"].item()),
        template_ridge_constructed_rows=int(cache["template_ridge_constructed_rows"].item()) if "template_ridge_constructed_rows" in cache.files else 0,
        template_ridge_feature_count=int(cache["template_ridge_feature_count"].item()) if "template_ridge_feature_count" in cache.files else 0,
        template_ridge_condition=float(cache["template_ridge_condition"].item()) if "template_ridge_condition" in cache.files else 0.0,
        template_ridge_trace_hat=float(cache["template_ridge_trace_hat"].item()) if "template_ridge_trace_hat" in cache.files else 0.0,
        template_ridge_profile_sha256=str(cache["template_ridge_profile_sha256"].item()) if "template_ridge_profile_sha256" in cache.files else "",
        template_ridge_alignment_event=str(cache["template_ridge_alignment_event"].item()) if "template_ridge_alignment_event" in cache.files else "",
        posterior_prior_variance=float(cache["posterior_prior_variance"].item()) if "posterior_prior_variance" in cache.files else 0.0,
        posterior_prior_sha256=str(cache["posterior_prior_sha256"].item()) if "posterior_prior_sha256" in cache.files else "",
        posterior_design_rank=int(cache["posterior_design_rank"].item()) if "posterior_design_rank" in cache.files else 0,
        posterior_design_condition=float(cache["posterior_design_condition"].item()) if "posterior_design_condition" in cache.files else 0.0,
        posterior_reliability_formula=str(cache["posterior_reliability_formula"].item()) if "posterior_reliability_formula" in cache.files else "",
        posterior_reliability_epsilon=float(cache["posterior_reliability_epsilon"].item()) if "posterior_reliability_epsilon" in cache.files else 0.0,
        posterior_reliability_zero_floor=float(cache["posterior_reliability_zero_floor"].item()) if "posterior_reliability_zero_floor" in cache.files else 0.0,
    )


def load_unit_side_features(
    nwb_path: Path,
    *,
    feature_group: str,
    pool_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    cache_dir: Path | None = None,
    permutation_seed: int | None = None,
    bin_size_ms: int = 20,
    window_size: int = 50,
    trial_result_filter: str = "R",
    signal_view: str = "sua",
    template_profile: np.ndarray | None = None,
    posterior_prior: dict[str, object] | None = None,
) -> tuple[np.ndarray, SideFeatureMetadata]:
    """Load normalized per-unit (SUA) or per-electrode (pseudo-MUA) features."""
    if feature_group not in KNOWN_FEATURE_GROUPS:
        raise ValueError(f"Unsupported feature_group {feature_group!r}")
    _validate_signal_view(signal_view)

    raw: np.ndarray
    metadata: SideFeatureMetadata
    compute_kwargs = {
        "feature_group": feature_group,
        "pool_size": pool_size,
        "bin_size_ms": bin_size_ms,
        "window_size": window_size,
        "trial_result_filter": trial_result_filter,
        "signal_view": signal_view,
        "template_profile": template_profile,
        "posterior_prior": posterior_prior,
    }
    if cache_dir is None:
        raw, metadata = compute_unit_side_features_uncached(nwb_path, **compute_kwargs)
    else:
        cache_path = _side_feature_cache_path(
            cache_dir,
            nwb_path,
            feature_group=feature_group,
            pool_size=pool_size,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            trial_result_filter=trial_result_filter,
            signal_view=signal_view,
            template_profile_hash=_array_sha256(template_profile) if template_profile is not None else None,
            posterior_prior_hash=str(posterior_prior["prior_sha256"]) if posterior_prior is not None else None,
        )
        with _exclusive_cache_lock(cache_path):
            if cache_path.is_file():
                try:
                    with np.load(cache_path, allow_pickle=False) as cache:
                        raw = cache["features"].astype(np.float32, copy=False)
                        metadata = _metadata_from_cache(cache)
                    logger.info("Loaded cached side features for %s from %s", nwb_path.name, cache_path)
                except (KeyError, OSError, ValueError) as exc:
                    logger.warning("Discarding unreadable side-feature cache %s: %s", cache_path, exc)
                    cache_path.unlink(missing_ok=True)
                    raw, metadata = compute_unit_side_features_uncached(nwb_path, **compute_kwargs)
                    _write_npz_atomically(cache_path, features=raw, **_metadata_to_cache_arrays(metadata))
            else:
                raw, metadata = compute_unit_side_features_uncached(nwb_path, **compute_kwargs)
                _write_npz_atomically(cache_path, features=raw, **_metadata_to_cache_arrays(metadata))
                logger.info("Cached side features for %s at %s", nwb_path.name, cache_path)

    normalized = ((raw - mean) / std).astype(np.float32)
    if permutation_seed is not None:
        normalized = permute_side_feature_rows(
            normalized, permutation_seed=permutation_seed
        )
    return normalized, metadata

def load_session_electrode_ids(nwb_path: Path) -> np.ndarray:
    """Per-sorted-unit NWB electrode indices for one session (not z-scored)."""
    with NWBHDF5IO(nwb_path, "r") as io:
        units_df = io.read().units.to_dataframe()
    return electrode_ids_from_units(units_df)


def permute_electrode_ids(
    electrode_ids: np.ndarray,
    *,
    permutation_seed: int,
) -> np.ndarray:
    """Return a copy of ``electrode_ids`` permuted along the unit axis."""
    generator = np.random.RandomState(permutation_seed)
    perm = generator.permutation(electrode_ids.shape[0])
    return electrode_ids[perm].astype(np.int64, copy=False)


def compute_electrode_vocab_size(nwb_paths: Sequence[Path]) -> int:
    """Largest electrode index + 1 across sessions (shared ``nn.Embedding`` vocabulary)."""
    max_id = -1
    for nwb_path in nwb_paths:
        ids = load_session_electrode_ids(nwb_path)
        if ids.size:
            max_id = max(max_id, int(ids.max()))
    if max_id < 0:
        raise ValueError("No electrode ids found across the supplied NWB files")
    return max_id + 1


def side_feature_stats_sha256(mean: np.ndarray, std: np.ndarray) -> str:
    payload = {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
