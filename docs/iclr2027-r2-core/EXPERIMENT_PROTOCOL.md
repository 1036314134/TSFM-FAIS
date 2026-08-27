# B-FAIS ICLR 修订实验协议（R2）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-06
- Verification Status: UNVERIFIED
- Version Label: bfais_experiment_protocol_r2_v1

本文件在查看 R2 新结果前固定实验问题、数据划分、随机性、评价指标和解释规则。旧结果版本定义为 `legacy-rolling-20260806-v1`；R2 候选版本使用 `iclr27-r2-candidate-vNNN`。两版结果独立保存，论文中的同一张表或图不得混用两版数字。

## 研究问题与对照边界

R2 首先检验四个核心问题：下游预测监督是否优于仅依据重构误差的内部控制；块级选择是否优于序列级选择；结构化块路由是否优于相互独立的块路由；路由器能否迁移到训练标签中完全未见的数据族和预测模型。

预测误差只用于 B-FAIS 及其内部控制版本。MetaOD、DSelect1、NeuralUCB、ALORS、Hybrid-LSTM 和随机选择器保持各自原始训练目标、输入与实现，不向它们提供预测损失。所有方法共享相同的候选填补池、item、预测起点、缺失 mask 和预测采样，避免把样本差异误记为方法差异。

内部控制固定为 `Seq-Recon`、`Seq-Forecast`、`Independent-Block-Forecast`、`Structured-Block-Forecast`。其中 Forecast 与 Recon 的区别仅为训练监督；Sequence 与 Block 的区别仅为选择粒度；Independent 与 Structured 的区别仅为块关系、图和协调求解。若实现无法满足单变量差异原则，该比较不得进入核心结论。

## 数据、划分与可用信息

ETT 家族仅用于开发、阈值选择、目标选择和超参数选择，不进入确认实验的主统计。数据清单另有 18 个 non-ETT 家族；`housing_inventory` 的序列长度 114，小于固定的 context 96 与 horizon 96 之和，预先判为不可评估并排除。`job_claims` 只有一个可用预测起点，rolling-origin 只能将其分配给训练，因此 rolling confirmation 固定为其余 16 个家族并按家族等权汇总。LOFO 可以把该唯一预测起点全部用于留出评估，所以 LOFO 主结果覆盖 17 个家族，同时另报排除 `job_claims` 的 common-16 敏感性结果。所有确认数据在结果生成后保持只读分析，不再用于修改超参数或回退阈值。

rolling-origin 实验继续使用每个数据集最前 20% 历史拟合本地候选填补器，并在其后的较早起点生成路由训练标签、较晚起点评估。LOFO 主实验对每个非 ETT 数据族逐族留出：留出族不得参与路由器、先验、校准、阈值或教师标签训练；允许其最前 20% 历史仅用于无下游标签的候选填补器拟合。严格零样本候选迁移属于 P1，不作为本轮主要主张。

特征策略固定为三档。`legacy` 保持现有输入，仅用于旧设置复现。`deployment_available` 删除 `dataset_id::*`、`family_id::*`、`missing_mechanism::*` 和由缺失生成器直接给出的 `target_missing_rate`，其余缺失率和块结构由当前可见 context 与 mask 计算。`identity_free` 在 `deployment_available` 基础上再删除 `forecast_model::*`，用于未见预测模型迁移；候选 ID 与候选家族仍保留，否则路由器无法区分候选行为。正式 Core 与 LOFO 使用 `deployment_available`，预测模型迁移使用 `identity_free`。

训练、开发和确认 mask seed 互不相交，分别固定为 `[1101, 1102, 1103]`、`[2101, 2102, 2103]` 和 `[3101, 3102, 3103]`。每个数据集、缺失机制和缺失率至少产生三个独立 mask；六种机制与五个缺失率合计至少 90 个 episode/数据集/预测模型。若数据长度不足，应在清单中记录实际 episode 数，不以复制样本补足。

路由训练 seed 固定为 `[4101, 4102, 4103, 4104, 4105]`。每个 seed 必须实际传入 LightGBM ranker 和 pairwise regressor，并记录在路由 manifest；不得只改变目录名。正式教师标签与下游评价预先固定使用 20 个 Chronos 预测样本，不以开发结果降低该数值。ETT 另做 5、10、20 个样本的敏感性分析，用于报告精度与计算量变化；该分析不再改变正式配置。仅对最终版本增加三个预测采样 seed。

## 模型与实验矩阵

正式已见预测模型为 Chronos-2 与 TimesFM 2.5，二者分别报告。第三个完全未见预测模型在查看迁移结果前固定为 Sundial `thuml/sundial-base-128m`，本地 snapshot 为 `3212e42564493f520593e5414af4367fc4b49226`。Sundial 不产生路由训练标签、不参与超参数与阈值选择，只评估由 Chronos-2 与 TimesFM 2.5 标签训练的 `identity_free` 通用路由器。

P0 顺序固定如下：P0-0 冻结并重算 Legacy 的非 ETT 与尾部指标；P0-1 在 ETT 上比较 `full_candidate_loss`、`forecast_loss`、`routing_target`；P0-2 比较 Forecast 与 Recon；P0-3 比较 Sequence、Independent Block 与 Structured Block；P0-4 依次移除伪缺失证据、块关系、协调求解、预测一致性、全局/数据集先验、校准与收缩、身份特征和回退；P0-5 完成 17 个可评估 non-ETT 数据族 LOFO，并补充 common-16 敏感性汇总；P0-6 完成 mask 与路由独立重复；P0-7 分析尾部风险并验证开发集确定的回退；P0-8 评估 Sundial 迁移。

P1 在 P0 结论稳定后执行：基于真实故障几何的半合成缺失、真实自然缺失、48/96/192 长度敏感性、候选数与伪块数效率测试，以及同时未见数据族和预测模型的组合测试。若 `cost_weight=0` 且未实现真实成本项，论文删除成本优化相关表述。

## 指标、统计与成功判定

主指标为 MASE，Chronos-2 和 TimesFM 2.5 分开报告，非 ETT 数据族等权。辅助指标包括 sMAPE、填补误差、运行时间、预测调用次数和峰值显存。配对置信区间按 family → dataset → item → origin → mask 分层重采样，固定 5,000 次 bootstrap。核心比较使用预先列出的六组检验，并以 Holm 方法校正：Forecast 对 Recon；Block 对 Sequence；Structured 对 Independent；Core Full 对最强外部选择器；LOFO Core Full 对 LOFO Seq-Forecast；Sundial 迁移对稳健固定填补器与 Seq-Forecast。

主要优越性要求 family-macro 的配对差值 95% 置信区间上界小于 0，并且经校正的 p 值小于 0.05。平均指标不满足优越性时，不使用“显著改进”表述。预先可选的非劣界为相对 MASE +1%，仅用于方法简化判断，不能代替贡献验证。

尾部风险相对“仅使用训练数据选择的稳健固定填补器”定义，报告中位数、P90、P95、CVaR90、CVaR95、最大退化和超过预设退化阈值的 episode 比例。测试集事后最佳候选仅作为 oracle 诊断。回退阈值只在 ETT 上选择，目标为降低 CVaR90，且平均 MASE 相对无回退版本恶化不超过 2%；确认集不再调阈值。

## 结果解释规则

Forecast 稳定优于 Recon 才支持下游预测监督贡献；否则弱化或删除该主张。Block 稳定优于 Sequence 才支持块级选择；否则将方法范围收缩为预测感知的序列级选择。Structured 稳定优于 Independent 才保留块关系和协调模块；否则简化实现。LOFO 有效才主张未见数据族迁移；否则限定为已知数据源上的时间外推。Sundial 迁移有效才主张一定的跨预测模型能力；否则明确方法需要模型专属训练。回退未同时改善尾部和满足平均性能约束时，仅保留失败分析。

负面结果不会触发删除、覆盖或选择性隐藏。R2 效果较差时，`legacy-rolling-20260806-v1` 仍作为“已见数据集、已见预测模型、rolling-origin”设置下的可复现结果保留；新版论文相应收缩主张。旧设置和 R2 设置不得拼接成一组正式统计。

## 资源与产物规则

本机有其他 CUDA 作业时，不启动 `fit-imputers`、`labels`、含预测一致性的 `impute` 或 `evaluate`。GPU 阶段仅在阻塞作业父进程及其 CUDA 子进程结束，并连续三次、间隔约 10 秒满足 GPU 利用率低于 10% 和空闲显存不少于 9 GiB 后启动。CPU 阶段可执行配置验证、标签合并、LightGBM 路由训练、统计汇总和旧结果校验。

所有新运行写入 `artifacts/iclr27-r2/`，run ID 使用 `r2-{study}-{split}-{variant}-{model}-ms{mask_seed}-rs{router_seed}-{stage}-vNNN`。新目录不得复用 `main-seq96-opt*`、不得对 Legacy 目录使用 `--resume`、不得原位覆盖失败运行。每个运行记录代码 commit、dirty 状态、完整配置及 SHA-256、数据与预测模型 snapshot、上游产物签名、全部 seed、预期/实际 episode 数、状态与失败原因。
