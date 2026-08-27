# E05 Rotation Consistency

## Status

Completed and audited on 2026-08-25. The implementation is on
`exp/e05-rotation-consistency` at `f2bd586`.

## Protocol

Frozen E01 T4/B3S seed-42 diagnostic only: 37 train and 8 validation
sessions, 64 deterministic windows per session, and 32 seeded random SO(2)
angles plus zero. Both physical-pipeline and normalized-internal rotations were
measured. Formal test sessions were excluded; optimizer, backward, and weight
updates were disabled. The student state SHA-256 was identical before and
after evaluation.

## Results

The following values exclude the zero-angle identity control.

| Split | Rotation path | Mean epsilon | Mean relative RMS | Worst epsilon | Worst relative RMS |
|---|---|---:|---:|---:|---:|
| train | physical pipeline | 2.539907 | 0.631671 | 7.477787 | 1.219930 |
| train | normalized internal | 2.508826 | 0.626788 | 7.105257 | 1.158350 |
| validation | physical pipeline | 1.283444 | 0.410934 | 2.528500 | 0.661792 |
| validation | normalized internal | 1.318310 | 0.437711 | 2.577187 | 0.716864 |

Audit receipt: `pass=true`; 45 allowed sessions, 8 excluded test sessions,
33 angles, 64 windows/session, and 2,970 raw rows.

## Conclusion

The current T4 consumer has a substantial task-frame SO(2) violation on both
source and unseen validation sessions. E05 therefore supports proceeding to
E06; it does not by itself establish that hard equivariance improves external
R2.

Raw artifacts:

- `sua_exploration/results/e05_rotation_consistency_t4_v1/E05_ROTATION_CONSISTENCY.json`
- `sua_exploration/results/e05_rotation_consistency_t4_v1/E05_ROTATION_CONSISTENCY_AUDIT.json`
- `logs/e05_rotation_consistency_t4_seed42_restart1.log`
