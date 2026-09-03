# H1 CAL-AUG M3 Diagnostics V1 A1

The first immutable attempt stopped before metrics because FalconEvaluator
right-pads shorter recordings within its multi-recording batch, while raw NWB
TrialNum has the unpadded length. A1 retains every scientific and evaluation
rule and only right-pads TrialNum with NaN to the evaluator timeline. Since
eval-mask is false on padded rows and NaN never matches a calibration trial,
padding cannot enter reuse scoring. A1 uses a fresh result root and does not
retry or modify the failed root.
