# Amendment: H1 CAL-AUG M3-Aware Dual-Selection V2 Evaluation A1

The V2 A1 C2 training completed all 50 epochs and passed full training integrity. Its offline evaluation then failed before inference because the evaluation layer reused an M4-specific held-out record helper that requires at least five legal trials. The registered held-out-calib recordings have exactly three legal trials, which is valid for the preregistered HO-M3 development/model-selection surface but invalid for an M4 post-calibration query protocol.

This additive amendment leaves the failed root, all checkpoints, source authority, schedule, plan, normalizer, and training receipts unchanged. It introduces a dedicated M3-compatible held-out record loader that accepts at least three chronological eval-valid TrialNum values and never calls the M4 support/query gate.

The amendment is evaluation-only. It SHA-binds the passed A1 training integrity receipt and all 50 C2 checkpoints, plus frozen V1 C1 epoch49. It constructs earliest-M3 identity/carrier payloads with the unchanged V1 plan and `s_src`, then evaluates HI-M3 on GPU0 and HO-M3 on GPU1 in parallel. It performs no training, optimizer step, backward pass, model update, checkpoint modification, Docker build, or EvalAI submission.

HO-M3 remains a development/model-selection surface that scores held-out-calib data used for M3 calibration. It is not an untouched held-out generalization metric. Selection rules and all final reporting fields remain unchanged from the V2 work order.
