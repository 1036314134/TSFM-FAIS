# B-FAIS R2 运行手册

本手册服从 [EXPERIMENT_PROTOCOL.md](./EXPERIMENT_PROTOCOL.md)。命令只写入 `artifacts/iclr27-r2/` 或版本化备份目录；不得向现有 `main-*` 产物写入。

## 阶段顺序

G0 固化 `legacy-rolling-20260806-v1`，逐文件校验源端和备份端 SHA-256。G1 验证全部 R2 配置并运行单元测试。G2 用既有标签完成实现冒烟测试，并完成 P0-0 Legacy 重分析；既有标签不得充当 R2 正式证据。G3 在 GPU 释放后生成新的 train/development/confirmation 标签。G4 使用新标签完成 P0-1 目标比较、标签合并和五个路由 seed 的 CPU 训练。G5 运行 impute 与 evaluate。G6 生成分层 bootstrap、Holm 校正、尾部统计和完整性报告。只有一个阶段的 manifest 为 `completed` 且输出计数与哈希通过时，后继阶段才可启动。

## 当前资源门槛

GPU 作业启动前需要同时确认：另一实验的调度父进程及其 CUDA 子进程均结束；连续三次、间隔约 10 秒的 GPU 利用率均低于 10%；空闲显存不少于 9 GiB；没有新的计算型 Python CUDA 进程。运行中保留至少 1.5 GiB 显存余量。CUDA OOM、Xid、温度达到 80°C 或产物超过 10 分钟无增长时记录告警并暂停人工检查；除明确硬超时外不自动终止。

## 首批命令原则

配置验证、统计和 `train-router` 可在当前 GPU 忙碌时执行。正式 GPU 命令在运行前记录完整命令、工作目录、预期输出、硬超时、日志路径和监测文件。首次标签运行按单一预测模型启动，使用新的 `r2-` run ID；发生异常时保留失败目录并换用下一个 `vNNN`，不静默重试。

P0 可复用已经冻结且协议一致的候选填补器 artifact，以避免无必要的深度候选重训；复用前必须校验数据、context/horizon、前 20% 拟合范围、候选池和模型文件哈希。当前 [复用审计](./IMPUTER_REUSE_AUDIT.md) 已确认 Legacy 与 R2 的根 seed、训练 mask 和数据族范围不同，因此首批 R2 必须重新运行 `fit-imputers`，不得把旧候选产物当作新协议产物。

rolling non-ETT 训练固定排除 `ett` 与不可评估的 `housing_inventory`，覆盖 17 个家族；rolling confirmation 再排除没有后续评估起点的 `job_claims`，覆盖 common-16。LOFO 覆盖 17 个家族，并额外输出 common-16 汇总。正式标签和评价固定使用 20 个预测样本。目标审计的三份 router 固定只使用各自 unary ranker 风险，关闭原始 proxy 分数与全局先验的额外混合；目标选定后的完整系统再单独确定证据组合。

## 完成检查

每个阶段检查 `stage_manifest.json` 的状态、配置摘要、输入签名和执行标记；labels 检查 unary、pair、group 与 episode 计数；router 检查实际 seed、训练目标、特征策略和留出对象；impute/evaluate 检查每个 dataset/item/origin/mask/method 的完整配对；统计检查 ETT 排除、family 等权、两模型分开、5,000 次分层 bootstrap、核心比较范围和 Holm 校正。缺失任何检查项时，候选版本保持 `UNVERIFIED`。
