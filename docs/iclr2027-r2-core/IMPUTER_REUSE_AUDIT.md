# R2 候选填补器复用审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-06
- Verification Status: VERIFIED
- Version Label: bfais_imputer_reuse_audit_r2_v1

本审计比较 Legacy 候选填补器产物 `artifacts/main-seq96-opt13-fit-v1/imputer_artifacts` 与首批 R2 non-ETT 训练配置 `configs/iclr27-r2/target-audit/full_candidate_non_ett_train.yaml`。结论是正式 R2 不复用该候选产物，必须生成新的 `fit-imputers` 运行。旧产物保持只读，仅用于 Legacy 复现。

两者保持一致的设置包括 `context_length=96`、`horizon=96`、`fit_prefix_fraction=0.2`、`training_window_stride=24`、缺失块长度、六种缺失机制、五个缺失率、全部候选、每数据集最多四个 item、最多 64 个拟合窗口、深度候选 10 个 epoch、batch size 16、CSDI 样本数 5 和 MissForest 并行数 8。

影响拟合输入的设置存在三项不一致。Legacy 根 seed 为 `20260710`，R2 根 seed 为 `20260806`，这会改变每数据集的确定性 item 子集。Legacy 拟合 mask seed 为 `[20260710]`，R2 为 `[1101, 1102, 1103]`，这会改变合成缺失训练窗口。Legacy 产物覆盖 32 个启用数据集并包含 ETT，首批 R2 non-ETT 运行排除 ETT 以及长度不足固定 context 与 horizon 的 `housing_inventory`，覆盖 27 个启用数据集。配置中的 split 也从 `leave_family_out` 变为 `rolling_origin`；尽管该字段不直接决定候选拟合器的目标函数，完整拟合身份已经不同。

输入签名如下：Legacy `resolved_config.json` 为 `bd7e7923dbeb38e6d0c403bfcf483ec75b13d1dced70c8bdb3f53c09328c49a6`，Legacy `imputer_artifacts/manifest.json` 为 `3bb942ab5b6f4148c80adf35ab7b5de9d9fe9d453cd8c8655934b5434c5cf124`，数据审计文件为 `c7c361b2a2532ffe66abdd5af2377c3ea1aeb7693a021c3b62aaa3b75b69a3fa`，数据清单为 `c5603df7e77f2d7a5b80f86ca49fe2e35d87d624f4ea3bcf5451db76186d97c6`，候选池清单为 `7743eca38f9f8fb6304690563f9cc65869345b70eb966319217fab4f77adc1e2`，R2 训练配置为 `7545f5af02ecb324a1a7eecd478f9e013c71a0920bb1ffcec6ef14461f8ce96e`。

因此，资源门控解除后首先运行 R2 non-ETT `fit-imputers`。后续 Chronos-2 与 TimesFM 2.5 教师标签只能引用该新产物；任何指向 `main-seq96-opt13-fit-v1` 的正式 R2 标签命令均视为协议不兼容。
