# R2 external sequence-selector configurations

These wrappers keep the external sequence selectors on the objective implemented by `configs/router/baseline_selector_suite.yaml`. Their registered target is `masked_context_reconstruction_asmape_v1`, which is mapped to `ranker_target: imputation_loss`; downstream forecast loss is not supplied as a training target to MetaOD, DSelect1, NeuralUCB, ALORS, Hybrid-LSTM, or the random selector.

`rolling_non_ett_train.yaml` trains on the 17 eligible non-ETT families. `ett_development_eval.yaml` is limited to ETT development data. `rolling_common16_confirmation_eval.yaml` removes `job_claims` in addition to ETT and `housing_inventory`, leaving the common 16-family confirmation set. `lofo17_train.yaml` and `lofo17_eval.yaml` use the same 17 eligible non-ETT families with family-wise holdout.

All five configurations fix root seed `20260806`, router seed `4101`, `deployment_available` features, 20 forecast samples, a 90-episode-per-dataset cap for the active partition, isolated R2 output under `artifacts/iclr27-r2`, and Sundial revision `3212e42564493f520593e5414af4367fc4b49226`. Their `teacher_forecaster_ids` lists are empty because these selectors receive only the imputation-quality label identity. Training, development, and confirmation masks use disjoint seed sets `1101..1103`, `2101..2103`, and `3101..3103`, respectively.
