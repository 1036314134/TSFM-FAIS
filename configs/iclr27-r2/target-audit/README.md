# ICLR27 R2 target audit configs

本目录冻结三种 B-FAIS 排序目标的第一批对照配置。三组配置除排序目标及其 router YAML 外保持相同的数据、候选、特征、mask 和 router seed 设置。

目标审计的三份 router 配置均固定 `r1=1`，并将 `r0`、伪缺失 proxy 和 `global_prior` 权重设为 0。这样，推理风险只来自当前被比较目标训练出的 unary ranker，不会让 `forecast_loss` 或 `routing_target` 版本再次借用由 `full_candidate_loss` 计算的全局先验。最终完整系统的证据组合在目标选定后单独使用 ETT 开发集确定，不与本目标审计混用。

| 目标 | Router 配置 | 训练 | ETT development | non-ETT confirmation |
|---|---|---|---|---|
| 完整候选预测损失 | `block_fais_full_candidate.yaml` | `full_candidate_non_ett_train.yaml` | `full_candidate_ett_development.yaml` | `full_candidate_non_ett_confirmation.yaml` |
| 单块反事实预测损失 | `block_fais_block_local.yaml` | `block_local_non_ett_train.yaml` | `block_local_ett_development.yaml` | `block_local_non_ett_confirmation.yaml` |
| 协调后的块级目标 | `block_fais_routing_target.yaml` | `routing_target_non_ett_train.yaml` | `routing_target_ett_development.yaml` | `routing_target_non_ett_confirmation.yaml` |

教师标签只生成一次。使用 `full_candidate_non_ett_train.yaml` 完成候选拟合及 Chronos-2、TimesFM 2.5 标签生成并合并。该标签文件同时包含 `full_candidate_loss`、`forecast_loss` 和 `routing_target`。随后，三个 `*_non_ett_train.yaml` 仅用于 `train-router` 阶段，并引用同一份新合并标签。不得为另外两种目标重新生成教师标签，否则会把标签采样差异混入目标比较。

训练配置排除 `ett` 和长度不足 192 的 `housing_inventory`，保留 17 个可训练 non-ETT 家族并使用 train mask seeds `1101..1103`。development 配置只包含 `ett`，使用 `2101..2103`。rolling confirmation 进一步排除只有一个可用预测起点的 `job_claims`，固定为 16 个可确认家族并使用 `3101..3103`。全部配置使用数据采样 seed `20260806`、router seed `4101`、20 个正式预测样本、每数据集最多 90 个 episode、`deployment_available` 特征策略，以及 `artifacts/iclr27-r2` 输出根目录。

每个 stage 必须使用新的 `r2-` run ID。Router seed 重复只修改 `experiment.router_seed` 为预注册的 `4102..4105`；数据采样 seed、mask seeds、标签文件和其他配置保持不变。
