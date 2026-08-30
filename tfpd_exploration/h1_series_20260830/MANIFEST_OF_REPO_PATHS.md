# H1-related repo paths (not duplicated in this bundle)

## Training trees (in-repo)
- `SPINT-main/src/` — the H1 FALCON training modules/contracts
  (h1_m4_eb_pilot_contract.py, h1_m4_eb_normalized_v2_contract.py,
  h1_m4_cce_contract.py, models/h1_carrierid_all_source_official_module.py,
  src/data/h1_m4_eb_pilot.py, ~1070 H1-named files incl. configs)
- `streaming_calibration_exp/src/` — h1_clean_nested_loso_eval.py,
  h1_lfmc4.py; the FalconLitModule decode path
  (`src/models/streaming_calibration_module.py:479-489`) and the generic
  loader pre-history padding (`src/data/falcon_datamodule.py:213-234`) that
  the mask contract (docs/DESIGN_H1_WINDOW_MASK_CONTRACT_V1_20260830.md)
  targets

## Large artifacts (regenerate, do not push)
- `tfpd_exploration/results/h1_date_lodo_checkpoint_cache_v1/` (1.6 GB)
- `tfpd_exploration/results/h1_variable_activity_exposure_v1/checkpoint.pt`
  (JSON receipts ARE in this bundle)

## Sealed lineage docs
- `tfpd_exploration/docs/RESULTS_QUEUE_20260829_AC3U_SLOT_AUDIT_AFFINE.md`
  (the CAL-AUG mechanism result motivating task 4)
