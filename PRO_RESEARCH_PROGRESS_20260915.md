# TSFM-FAIS 仓库进展与研究交接

快照日期：2026 年 9 月 15 日，北京时间 11:06。仓库位置：`E:/ZMY/Github/TSFM-FAIS`。本文面向后续 Pro 研究判断，汇总研究问题、方法演变、已完成证据、失败结果、工程与稿件状态，以及需要优先回答的问题。本文可单独阅读；文中路径用于回到本地证据。实验数字转述自仓库报告和已有审查记录，本次整理没有重新训练、重新评分或重跑全量测试。R15 的状态另外核对了实际队列、错误日志和进程；整理期间出现了新的 R15 修复说明，已补入下文。

## 1. 当前结论

项目最初研究“为不同缺失块选择填补算法以改善冻结时序预测器的预测”，后来逐步转向“先获得多种填补对应的预测，再学习选择或组合这些预测”。截至本快照，R2—R14 已积累多轮实验与诊断，尚没有在两个主要预测器、真实缺失面板及强固定组合/预测中位数之间成立的稳定方法优势。仓库最新研究报告仍判断为 NOT READY，不能把工程完成、源开发改善或模拟诊断完成理解为投稿就绪。

目前最清楚的经验事实是：若干学习方法在源开发或部分人工缺失面板上有改善，但新确认或原始缺失数据上的收益不稳定；七/八候选预测中位数、源固定凸组合、原生缺失处理和长历史输入都是必须保留的强对照。源样本扩张、内部表示、时间校准、损失调整、目标级特征、候选池对齐和真实缺失源训练均已经做过，不能将这些方向笼统列成“尚未尝试”。现有结果也没有证明所有选择器无效或真实数据中的选择收益不可学习。

当前最新分支为 R14/R15：在已知线性高斯生成过程下，区分可用历史条件下的期望风险收益与看过真实未来之后的事后最优收益。R14 已通过独立审查；R15 已生成全部读出；首次独立审查失败后，数组排列修复版 v002 已通过完整审查，研究结果尚待综合解读。该分支研究的是评价性质，目前没有提升真实数据预测精度。

README 首部更新停留在 R5，正文大量内容仍是 7 月历史实验。其“当前状态”、候选覆盖、目录数量、测试数量和资源优先级均不应直接当作 9 月 15 日状态。主英文工作稿推进到 R6，最新 R7—R15 进展主要分散在单独报告中。只阅读 README 或现有论文 PDF 会漏掉关键后续结果。

## 2. 问题、方法与信息条件

输入是带缺失的多变量历史，输出是冻结时间序列预测器的未来预测。早期 B-FAIS 将每个变量上的极大连续缺失区间当作缺失块，通过单块风险、块间关系和搜索组装一段完整历史。R3 之后转向整序列收益估计与预测响应特征；R5/R6 重点转向预测组合，其输出通常不对应唯一的填补序列。

R6 代表性方法使用七个候选：LOCF、线性插值、季节滞后、多变量 KNN、SAITS、TimeMixer++，以及预测器自身的缺失输入处理。MoTM 起初作为额外对照加入预测中位数，R12 起也纳入同池学习器。早期 B-FAIS 的 20 个注册候选与这个七/八候选研究池是不同设置，不能混写。早期 TRMF 因适配生命周期与冻结产物协议不兼容而禁用。

给定候选预测 p_a，R6 输出为 sum_a w_a p_a，权重非负且和为一。Chronos-2 原设置联合读取多变量历史、两个目标共享权重；TimesFM 2.5 分别处理目标、分别产生权重。R11 专门比较了 Chronos 的目标级特征。对填补结果加权后调用非线性预测器，与直接对预测加权，是两个不同的动作空间。

R6 小网络每候选读取 33 维特征：21 个历史/补全特征和 12 个预测响应特征，其中点预测版本的两个分位数字段恒为零。共享编码器宽度为 16，每个种子 1,096 个参数，平均三个固定种子 5101/5102/5103 的权重。原日程为 25 轮、批量 128、AdamW 学习率与权重衰减均为 0.001。其主监督来自完整源历史经过同一冻结预测器得到的教师预测；匹配控制使用成员教师误差或源真实未来。后续轮次改变了部分条件，这些参数仅描述 R6 基准版本。来源：`docs/iclr2027/R6_METHOD_EXPLANATION_ZH.md`。

完整历史教师只在源训练或标明的理想参照中使用，部署时不能读取当前隐藏历史或未来。教师是预测器输出，不能直接视为真实未来的条件均值。目标前缀校准可以使用预测时刻之前的数据，但需要明确额外历史、标签和计算预算。使用真实目标未来拟合的每序列权重只能作为事后参照。

预测响应门控需要先取得最多七/八种候选历史的预测，再决定权重；批处理和相同输入缓存复用可以减少实际计算，不能将其称为单次预测调用的低成本选择。可微输入组合曾探索部署一次预测的路线，但目前没有形成优于中位数的证据。

当前主要指标为按训练前缀原观测均值和总体标准差标准化的下游 MAE/MSE，数值越低越好；填补重构误差仅辅助。R2/R3 的 MASE 不能与后续数字直接比较。原始未来未知值不填成真值，只有原可观测位置参与评分；R6 中 H96/H192 每目标分别至少要求 48/96 个可观测未来值。先标准化预测器输入与只标准化评分也是不同条件。跨轮表格必须核对面板、样本、聚合方式、输入处理、候选池和版本，不能仅凭同名方法直接比较。

## 3. 数据与独立证据规模

| 面板 | 已记录规模 | 正确解释 |
|---|---|---|
| 7 月旧主实验 | 32 个审计版本中 30 个进入评价，17 家族，900 episode/预测器 | 早期滚动实验，不能代替 R2 后注册检验 |
| R2 rolling 确认 | 26 个版本、16 个非 ETT 家族，2,340 配对 episode/预测器 | 5 路由种子、5,000 次分层重采样、Holm 校正；6 项注册检验均未满足优越性规则 |
| R3—R6 源开发 | 15 家族，165 个训练历史、52 个验证历史；5,940/1,872 遮盖任务 | 原展开含 6 机制×3 缺失率×2 掩码种子；同历史多掩码不能当独立历史 |
| R5 首轮自然确认 | 9 家族、18 序列、373 窗口，其中 7 家族的 164 窗口有原始历史缺失 | 已查看并用于后续开发，不能再次当未用确认集 |
| R5 跟进确认 | 823 任务、263 起点、39 序列 | 包含已见家族新序列、新 UCI 原生窗口和 weather/Solar 人工缺失，分面板报告 |
| R6 人工缺失 | 4 来源家族、32 个完整历史、1,152 遮盖输入，H96/H192 | 遮盖输入数不是独立历史数；这些数据后来已反复用于开发 |
| R6 北京原始缺失 | 192 缺失历史，另有 128 完整历史控制，12 站点 | 站点属于同一来源家族；完整输入不能稀释缺失子集 |
| R7 源表示采集 | 3,906 输入，15 家族、52 验证历史 | 本轮一个掩码种子，三个学习器种子不增加独立样本 |
| R9 源扩张 | 165→322 个训练历史，增加 157 历史、2,826 输入 | 只在 8 个长度足够的家族扩张；原 52 验证历史不变 |
| R14/R15 模拟 | 36 独立历史、180 观测输入；每条件 512 未来副本 | R15 完全复用 R14 历史；未来副本、相关系数、目标不增加独立历史 |

Bike/Occupancy 的未知时间格点单独报告，不自动解释为传感器故障。人工缺失与原始缺失、已见家族新序列与未见来源必须区分。历史确认数据和大部分开发数据现已被反复查看；下一轮若要证明新的方法优势，需要重新确定未使用的确认来源，不能把旧数据重划分后增加“独立确认次数”。

## 4. 研究路线与阶段结果

### 4.1 早期 B-FAIS、R2、R3 与 R4

7 月 B-FAIS 在两个预测器均有 16/30 数据版本严格胜出，非 ETT 为 12/26。这是旧口径下的胜出数量，后续 R2 没有支持更强的机制贡献。R2 历史稿记录：预测监督相对重构监督、块级相对序列级、图协调、未见家族和留出 Sundial 迁移均未满足注册优越性要求；6 项检验中 4 项完整但区间跨零、2 项因完整性条件不满足而不可用。北京 72 个自然缺失 episode 上，B-FAIS 在 Chronos/TimesFM/Sundial 均差于同人口可完整执行的最佳固定候选。来源：`docs/iclr2027/tsfm_fais_iclr2027.tex` 和 `docs/iclr2027-r2-core/EXPERIMENT_PROTOCOL.md`。

R3 完成整序列收益学习、静态/响应特征、参考动作、L1/L2、有界收益、历史反馈及默认缺失处理对照。扩大开发的响应选择 MASE 为 Chronos 3.304297、TimesFM 1.727528；对应中位数为 3.172805、1.752463，模型间方向不同。TimesFM 官方接口支持去除开头 NaN 并插值剩余 NaN，早期将它记为“不支持直接缺失”已更正。Chronos 输出布局和全缺失目标尺度也有专门核查，旧缓存与更正版本需要分开。来源：`R3_PROGRESS.md`。

R4 将主指标转向标准化 MAE/MSE，并展开信息预算与输入尺度检查。90 任务小面板中，1024 步直接历史相对 96 步的 Chronos MAE/MSE 下降约 15.82%/18.21%，TimesFM 约 12.52%/16.27%；它使用更多历史，且部分家族退化，不能算填补选择收益。Chronos 原始单位直接缺失输入的极端误差主要与两个全缺失目标任务相关，前缀尺度信息改变了比较。近期反馈、输入组合与输出组合没有稳定超过预测中位数。来源：`R4_SCREENING_FINDINGS.md`、`R4_SCALE_INFORMATION_NOTE.md`。

### 4.2 R5：教师、三候选组合与两次确认

可微块级输入组合已实现并运行五个版本，未超过同任务中位数，因此暂不采用为主方法。预测前学生、预测投影、结构化学生、预测响应教师、三元组目标和多种一致性/预算诊断均有代码与专门记录；这些属于探索分支，不应把存在计划文件自动写成全部已成功验证。

完整源历史教师排序后取前三预测的方法，在首轮 373 窗口确认上没有复现开发优势。以下仅是首轮原始历史缺失 164 窗口的标准化结果：

| 方法 | Chronos MAE/MSE | TimesFM MAE/MSE |
|---|---:|---:|
| 教师排序三候选 | 0.612649 / 0.812422 | 0.622398 / 0.826889 |
| 七候选预测中位数 | 0.609736 / 0.807308 | 0.614370 / 0.803062 |

43 选项回归未超过原方法。随后固定 35 个三元组、40 维特征和同配对学习器，直接监督实际组合教师误差的开发结果为 Chronos 0.772412/1.541547、TimesFM 0.758966/1.515527，均优于对应开发中位数。但跟进确认中，Chronos 新人工缺失相对中位数改善 6.84%/13.68%，与源固定三元组仍有 MAE/MSE 取舍；TimesFM 同面板退化 2.52%/2.69%，新自然缺失也没有一致收益。来源：`R5_RESEARCH_DECISION_20260913.md`、`R5_MATCHED_PORTFOLIO_RESULTS.md`、`R5_FOLLOWUP_RESULTS.md`。

神经填补预算已经比较 10 轮/64 窗口、50 轮/64 窗口、50 轮/512 窗口，并扩展到三个预测器、三个数据集各四个历史，共 12 历史/72 掩码任务。ETTh1 的三模型平均随较大预算改善，current_velocity_H 与 electricity 的三模型平均退化；最初电力 TimeMixer++ 对 Chronos 的改善只出现在四个历史中的一个。TiRex 在该预算分支有真实预测实验，不能沿用旧 README 的“仅有适配接口”。来源：`R5_BUDGET_ORIGIN_EXTENSION.md`。

### 4.3 R6：连续凸组合及其控制

R6 完成 23 方法、两个预测器、H96/H192 确认，重建核对 140,024 个预测向量和 27,396 个门控决策。主要 H96 人工缺失结果如下，来源为 `R6_CONFIRMATION_RESULTS.md`，原始证据版本是 `policy-results-v002`、`policy-audit-v002`、`readout-v002`。

| 方法 | Chronos MAE/MSE | TimesFM MAE/MSE |
|---|---:|---:|
| 预定组合教师主门控 | 0.608321 / 1.110985 | 0.624733 / 1.158096 |
| 成员教师目标控制 | 0.593649 / 1.134564 | 0.605122 / 1.132753 |
| 源真实未来监督控制 | 0.616020 / 1.147136 | 0.600364 / 1.130239 |
| 源固定凸组合 | 0.600751 / 1.109923 | 0.616414 / 1.148950 |
| 七候选中位数 | 0.614601 / 1.180633 | 0.619774 / 1.215528 |
| 八候选中位数，含 MoTM | 0.605507 / 1.139357 | 0.611607 / 1.171881 |

主方法相对七候选中位数的 MSE 有改善，但 MAE 不一致；相对源固定凸组合，两模型双指标均略差。北京原始缺失也未形成一致收益。TimesFM 的成员教师目标和真实未来控制优于预定主方法，不能事后改名成原主方法来回避原确认失败。

后续 R6 已完成教师迁移、逐坐标教师、动作类别、目标前缀校准、历史加权、源未来监督、旧自然缺失扩展、候选关系、逐位置组合、原生缺失源扩展和区间输入等检查。目标前缀教师校准改善 TimesFM、退化 Chronos；几何关系与逐位置输出在部分对照上改善但没有统一超越强静态控制；恢复两个区间字段只得到小幅指标取舍。完整历史教师或真实未来理想组合显示的空间不能当部署精度。详见附录文档索引对应 R6 条目。

### 4.4 R7—R13：逐项排查门控迁移问题

| 轮次 | 改动与已完成工作 | 实际结论 |
|---|---|---|
| R7 | 提取冻结预测器内部表示，384 学习器；另做同家族后续时间验证 | 主内部表示相对点特征在两模型双指标退化；训练误差降低、时间验证也退化，不能只归因于跨家族差异 |
| R8 | 内层时间校准、选轮数、约束相对固定组合的偏离；543 检查点 | Chronos 有改善但未超过固定组合；TimesFM 主结果完全不变；主结果等于仅早停控制，不能归功于新增约束 |
| R9 | 训练历史 165→322，控制更新次数；源端与迁移审查完成 | 主源结果未双模型一致改善；旧真实缺失仍不及八候选中位数，单纯扩张未解决迁移 |
| R10 | 平滑 MAE 与 MAE/MSE 联合训练目标，同损失固定组合；源端及迁移完成 | TimesFM 人工缺失有较明确改善；Chronos 和旧真实缺失未统一改善 |
| R11 | Chronos 目标级特征与广播特征，容量/样本/更新匹配；TimesFM 精确复用 | 源端改善未稳定迁移；人工缺失不及目标固定组合，旧真实缺失也退化 |
| R12 | MoTM 加入学习器，使其与强中位数同为八候选；源端与迁移完成 | 旧真实缺失仍弱于同池中位数；此前失败不能主要解释为学习器少一个候选 |
| R13 | 同八候选与联合目标，加入其他家族真实缺失历史，当前组整体排除；控制更新次数 | 九家族和旧七家族均未形成双预测器统一优势，暂停继续同类门控参数扩展 |

上述结果有正面局部信号，但逐项排查并没有建立统一优越性。R10 主联合真实未来模型在 R6 H96 人工缺失的 TimesFM 结果为 0.582252/1.065021，相对同损失固定组合 0.613581/1.145462 约降低 5.1%/7.0%；Chronos 为 0.610105/1.121383，未超过固定组合 0.609543/1.109277。它已经是开发性复用，不能作为新确认。

以下统一列出“旧七家族、164 个原始缺失历史”的后期结果，用来直接查看真实缺失主障碍。这里每行指标均来自相应轮次报告；不能拿前述 R6 人工缺失表的数字作为同面板对照。

| 方法/版本 | Chronos MAE/MSE | TimesFM MAE/MSE |
|---|---:|---:|
| 同面板八候选中位数 | 0.609386 / 0.797358 | 0.613597 / 0.793853 |
| R9 expanded_future | 0.612685 / 0.804050 | 0.621022 / 0.803161 |
| R10 metric_joint_future | 0.612616 / 0.804634 | 0.618077 / 0.801955 |
| R11 scope_target；TimesFM 复用 R10 | 0.614951 / 0.810056 | 0.618077 / 0.801955 |
| R12 同八候选学习器 | 0.612652 / 0.800212 | 0.619109 / 0.800245 |
| R13 real_augmented | 0.612528 / 0.802277 | 0.618856 / 0.800643 |

R13 九家族主结果为 Chronos 0.667436/1.060946、TimesFM 0.670347/1.009253，匹配更新次数的纯源控制为 0.667522/1.058792、0.670266/1.006515；这些微小变化不支持新增真实缺失训练带来稳定收益。对应来源为 `R9_TRANSFER_RESULTS.md` 至 `R13_REAL_POOL_RESULTS.md`。

### 4.5 R14：条件风险与事后选优诊断

R14 在耦合 AR、独立 AR、周期 AR 三种已知过程下，用精确可计算的条件分布分析候选预测。36 独立历史产生 180 观测输入，两个预测器共 2,232 个不同有效预测输入，4,320 个决策条件通过独立审查；凸组合最优性残差最大 1.38e-12，指标重建最大差 1.78e-15。八候选、未来创新标准差乘子 1、三个过程和四个缺失条件等权时，得到以下 MSE 结果。

| 比较 | Chronos | TimesFM |
|---|---:|---:|
| 条件最优单候选相对同面板固定算法的可获取收益 | 0.009988 | 0.015869 |
| 单候选配对事后选优偏差 | 0.038799 | 0.059736 |
| 偏差 /（偏差＋可获取收益），均值之比 | 约 79.53% | 约 79.01% |
| 条件最优凸组合相对均值组合的可获取收益 | 0.215022 | 0.285388 |
| 凸组合配对事后偏差 | 0.050821 | 0.071564 |

约 79% 是特定模拟面板的均值之比，不是现实数据不可学习比例；过程内比例再平均是另一统计量。单候选和凸组合使用不同参照，不能将单候选比例外推到组合。周期过程偏差绝对量明显小于其他过程；完整观测时各候选预测相同，偏差为零。已知生成参数下的条件最优也不是已实现的可部署学习器。来源：`R14_CONDITIONAL_RISK_RESULTS.md`、`R14_CONDITIONAL_RISK_PROTOCOL.md`。

### 4.6 R15：初次审查失败，修复版已通过

R15 固定条件均值与每个坐标方差，仅把未来协方差改为 Sigma_rho = rho Sigma + (1-rho) diag(Sigma)，rho 为 0、0.5、1。任意只依赖相同历史的固定点预测的期望 MAE/MSE、风险排序和条件 MSE 最优组合不变，拟检查事后选优差距是否随依赖结构变化。复用 R14 的 36 历史、180 输入和全部预测，没有新填补拟合或预测器调用。干预后的未来不能直接称为原 AR 过程的平稳延续。

实际状态比现有运行文档更新：2026-09-15 10:34:48 读出写出完成清单，包含 3,240 决策行，清单自报 R14 共享读出最大差为零。10:34:58 队列在 `r15_audit_future_dependence` 以退出码 1 失败；约 10:59 检查没有发现命令行包含本仓库的 Python 进程。队列仍保存旧 PID 字段，不能据该字段判断进程存活。

直接错误是 `scripts/audit_future_dependence.py` 第 181 行的 `np.testing.assert_array_equal(win_mae, witness["winner_mae"])`：512 个最优候选编号中有 1 个不一致。错误中“最大绝对差 2”是候选编号差，不是 MAE 误差。没有发现成功的 `dependence-audit-v001/manifest.json`。本次整理未修改或重启该实验。

11:05 补充：整理期间仓库新增 `docs/iclr2027/R15_AUDIT_REPLAY_NOTE.md`。该说明报告已定位为数组排列导致的浮点求和顺序差异：TimesFM independent_ar3 历史 1、30% 随机缺失、七候选、目标 0、rho=0.5 的副本 206，读出中候选 0/2 的 MAE 同为 0.9380721578677962；旧审查分别为 0.9380721578677966/0.9380721578677963，改变了最小候选编号。按原切片顺序重放后该样例全部 512 个编号一致，未来样本和方差逐值相同。说明登记了 `audit_future_dependence_v2.py` 与 `dependence-audit-v002`，保留原阈值和原产物。11:06 进一步核对发现 `artifacts/iclr27-r15/dependence-audit-v002/manifest.json` 已为 completed：全部 3,240 决策及 1,080 组不变期望风险/权重通过，最优性残差最大 8.16e-10、指标重建差 5.11e-15、稠密协方差差 1.20e-14，与 R14 共享读出差为零。因而最新状态是修复版完整审查通过，旧队列 failed 属于首次失败终态。本快照未进一步汇总 rho 干预的效应大小，也未据此判断新颖性。

直接证据是 `artifacts/iclr27-r4/gpu-queue-v001/state.json`、同目录 `r15_audit_future_dependence.log`、`artifacts/iclr27-r15/worker.stderr.log` 和 `artifacts/iclr27-r15/dependence-v001/manifest.json`。`artifacts/iclr27-r15/conditional-risk-queue-completion.json` 实际保存的是 R14 完成状态，不能当成 R15 审查通过证据。R15 的四项单元测试及静态检查已通过，但没有替代此次真实产物审查。

## 5. 工程、产物与稿件状态

代码包含数据加载/审计/整段掩码与窗口切分、候选填补器统一训练推理接口、冻结预测器适配、路由与预测组合、缓存与续跑、家族/时间隔离、原观测值保护、真实缺失评分和独立重建检查。`src/tsfm_fais/routing/` 保存多数早中期方法实现，近期 R7—R15 有较多研究实现直接位于 `scripts/`。快照统计为 213 个 Python 脚本、120 个测试文件；不能沿用 README 的 8 个脚本说明，也不能由文件数推断测试通过数。

| 入口 | 内容 |
|---|---|
| `src/tsfm_fais/data/` | 数据与掩码、划分、审计 |
| `src/tsfm_fais/imputers/` | 经典、结构化、深度、MoTM 等填补适配 |
| `src/tsfm_fais/forecasting/` | Chronos、TimesFM、TiRex、Sundial 适配、预测与指标 |
| `src/tsfm_fais/routing/forecast_gate.py` | 预测门控核心之一 |
| `scripts/r6_policy_inputs.py`、`scripts/r6_runtime.py` | R6 可见特征与模型运行公共逻辑 |
| `scripts/train_metric_source_gates.py`、`scripts/train_motm_pool.py`、`scripts/train_real_pool.py` | R10/R12/R13 训练入口 |
| `scripts/conditional_future.py`、`scripts/readout_conditional_risk.py`、`scripts/audit_conditional_risk.py` | R14 生成过程、读出与审查 |
| `scripts/future_dependence.py`、`scripts/readout_future_dependence.py`、`scripts/audit_future_dependence.py` | R15 依赖干预与独立核验 |
| `scripts/resume_utility_when_idle.py` | 实验队列、资源检查、续跑 |
| `configs/iclr27-r3/priority_first_r*_jobs.json` | 后期各轮正式队列配置 |
| `docs/iclr2027/R4_AUTONOMOUS_CONTINUATION.md` | 跨轮累计运行记录，后段覆盖至 R15 |

最近 Git 提交为 `9ae8768`，2026-09-10 15:30:45 +0800，标题“新增实验计划”。本次文件新增之前，工作树有 16 个已跟踪文件修改、424 个未跟踪条目，包含大量 R4—R15 脚本、测试和文档。仅查看已提交代码或普通 `git diff` 会漏掉未跟踪进展。`artifacts/`、`checkpoints/`、数据与模型目录被忽略，远程仓库或新 checkout 不会自动带上本机证据。本次只新增本文，不提交、不移动、不清理原文件。

环境声明为 Python >=3.10,<3.12；近期队列使用 `D:/Programme/Anaconda/envs/TSFM/python.exe`。各模型和深度填补依赖拆成可选依赖组。本地数据路径由 `configs/data/*.yaml` 及各轮准备清单确定，模型权重也依赖本地路径，不能仅安装代码就假定可完整复现。历史运行记录中的硬件为 RTX 3060 12 GB。最新实际队列使用 `priority=first`；README 及早期报告仍写第三优先级，后续执行不能机械照抄旧恢复命令。运行记录另记每小时检查与 9 月 17 日晚内部截止，这些是仓库历史执行约定，本次未核验或修改应用中的自动化。

测试证据分散在各轮产物中。README 的 396/418、R3 的 512 等均为历史快照，不是当前全树测试结论。R15 `dependence-tests-v002/manifest.json` 记录 4 项测试以及 lint/format/syntax 通过；随后真实审查仍失败。代码测试、预测重建、统计解释和方法优势分别是不同层次，不能互相替代。

稿件已有早期 ICLR 2026 文件、R2 英文历史稿、R5 工作稿、R6 工作稿及图表/章节构建脚本。当前已找到 R6 `artifacts/iclr27-r6/manuscript-render-v004/tsfm_fais_r6_draft.pdf`，清单为 18 页、正文 8 页；`visual_review.json` 记录修改页检查、其余页与 v003 匹配及 632 个数值单元检查。它明确说明 R7 在单独报告中，排版检查不代表研究就绪；R8—R15 更不能假定已经进入该 PDF。后续研究判断应以各轮结果为准，再确定如何重构稿件。

## 6. 尚未解决的研究问题

方法路线的关键缺口是强基线之上的可迁移收益。现有局部正面结果不能同时满足双预测器、双指标、原始缺失和公平信息预算。继续更换同类门控特征或监督目标需要新的理由；R7—R13 已排查多种常见解释，单纯列出更多模型容量、更多训练历史或加入 MoTM 不足以构成下一步。

诊断路线的关键缺口是新意及真实任务连接。R14 的结果可以说明指定生成条件下的事后收益含有乐观偏差，却不能解释现有真实数据究竟有多少收益可学习。R15 已通过的核验进一步提供固定边际风险下的依赖结构干预证据；是否超出已有算法组合评价研究，以及能否导出可用的评价准则或选择方法，仍需要论证。

已有文献核对记录包括 TSI-Bench、CleanIMP、ImputeGAP、Time-Indexed Imputation、GIFT-Eval、FoundTS；R14 定位文档另外记录 Cameron/Hoos/Leyton-Brown 的 IJCAI 2016 算法组合评价偏差、Wagner 等图像质量算法选择与 SynTSBench。这里仅转述仓库阅读记录，没有重新检索文献。不能声称首次研究填补下游效果、首次发现事后最优乐观偏差、首次采用可控时序或首次使用组合误差分解。后续若以其中某个区别作为主贡献，需要直接核对原论文的完整相关方法与假设。

实验覆盖的关键缺口是独立确认资源和统计口径。源开发历史数量有限，后期多次修改已使用相同验证/真实缺失数据；多掩码、多种子、多预测向量和大量评分行不能替代独立家族/历史。R6 四个有目的选择家族上的重加权区间只描述来源敏感性，不能当总体显著性。旧实验使用的模型、候选数量和历史范围不一致，需要在选定研究问题后确定最小但公平的统一主表。

可复现交付的关键缺口是材料分散。README、稿件、运行记录和现场状态已经存在版本差；大量最新代码未跟踪，审查产物只在本机。文档整合和证据索引有价值，但它们不能补足方法或研究贡献本身的缺口。

## 7. 建议 Pro 优先回答的问题

请首先判断项目目前最有依据的研究对象：继续开发能超过强固定组合的方法，还是围绕可获取收益与事后收益差设计评价研究，或当前证据不足以支持这两条路线。判断应同时保留 R5/R6 确认失败、R10 TimesFM 的局部改善、R13 真实缺失负面结果和 R14 的受控正面现象；不要仅选择某一轮支持预期叙事。

对 R15，先阅读新修复说明和已完成的 v002 审查，再汇总 rho 干预的实际效应大小、过程差异、单候选与凸组合差异。修复解决了读出核验问题，没有自动解决研究贡献问题；是否继续新一轮，应由这份结果解读决定。

如果继续方法路线，请指出一个现有结果支持、尚未被 R7—R13 覆盖的具体假设，并说明为何预计它能迁移到真实缺失、需要哪种可部署信息、与固定组合和中位数如何公平比较，以及什么结果会使该路线停止。预测器先验表示、目标级权重、数据扩张和候选扩张都已尝试，若重新使用，需要明确不同之处和必要性。

如果继续诊断路线，请说明 R14/R15 相对已有事后最优偏差研究增加了什么，以及是否能形成具有实际用途的评价量、实验设计建议或可学习性判断。尤其需要区分“已知模型条件分布下的精确上界”与“真实数据上仅凭历史可估计的量”；不能从模拟比例直接推出真实数据不可学习份额。

请最后给出按优先级排列的最小后续计划，逐项写明研究问题、所需数据/信息、强控制、成功/停止标准及预计计算量；指出哪些旧分支应停止、哪些结果可保留为正文或附录，以及是否有足够未用数据完成新确认。如果认为当前不应继续追求原定投稿目标，请直接说明依据。不要用录用概率或未经证实的新颖性评价替代证据判断。

## 8. 优先证据入口

| 用途 | 本地路径 |
|---|---|
| 历史确认失败与路线变化 | `docs/iclr2027/R5_RESEARCH_DECISION_20260913.md` |
| R6 方法定义与独立确认 | `docs/iclr2027/R6_METHOD_EXPLANATION_ZH.md`、`R6_CONFIRMATION_RESULTS.md` |
| R7—R13 逐项改进与迁移 | 各轮 `*_RESULTS.md`，完整文件名见下方索引 |
| 最新诊断定位与有效结果 | `docs/iclr2027/R14_CONDITIONAL_RISK_POSITION.md`、`R14_CONDITIONAL_RISK_RESULTS.md` |
| 最新诊断协议与修复 | `docs/iclr2027/R15_DEPENDENCE_PROTOCOL.md` 、`R15_AUDIT_REPLAY_NOTE.md` 和 v002 审查清单 |
| R6 主结果与审查 | `artifacts/iclr27-r6/readout-v002/`、`policy-audit-v002/` |
| R10 迁移证据 | `artifacts/iclr27-r10/metric-transfer-v001/` |
| R11 迁移证据 | `artifacts/iclr27-r11/target-local-transfer-v002/` |
| R12 迁移证据 | `artifacts/iclr27-r12/motm-pool-transfer-v001/` |
| R13 真实缺失源训练 | `artifacts/iclr27-r13/real-source-audit-v001/` |
| R14 读出与审查 | `artifacts/iclr27-r14/conditional-risk-v001/`、`conditional-risk-audit-v001/` |
| R15 读出与修复版审查 | `artifacts/iclr27-r15/dependence-v001/`、`artifacts/iclr27-r15/dependence-audit-v002/` |

若 Pro 只能读取这一份文件，上文足以理解主要进展与当前障碍；需要独立核对数字时，应再提供对应报告、汇总 CSV 和审查清单，不能假定云端能访问本地 `artifacts/`。下方索引覆盖研究 Markdown 文档，便于有仓库访问能力的阅读者追溯；标题仅表示文档主题，不表示计划已经执行或结论已经通过。

## 附录：研究文档完整索引

以下按轮次列出 docs/iclr2027 中现有 R 系列 Markdown，以及 R2 协议目录。索引在本次快照生成，不对每一份历史计划逐项重新判定执行状态。

### R2

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027-r2-core/EXPERIMENT_PROTOCOL.md` | B-FAIS ICLR 修订实验协议（R2） |
| `docs/iclr2027-r2-core/IMPUTER_REUSE_AUDIT.md` | R2 候选填补器复用审计 |
| `docs/iclr2027-r2-core/RUNBOOK.md` | B-FAIS R2 运行手册 |

### R3

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R3_CONFIRMATION_PROTOCOL_DRAFT.md` | R3 确认实验协议草案 |
| `docs/iclr2027/R3_LITERATURE_NOTES.md` | R3 文献定位记录 |
| `docs/iclr2027/R3_METHOD_NOTES.md` | R3 方法开发记录 |
| `docs/iclr2027/R3_PROGRESS.md` | R3 工作记录 |
| `docs/iclr2027/R3_RESEARCH_PLAN.md` | R3：冻结预测器的序列级填补收益估计 |

### R4

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R4_AUTONOMOUS_CONTINUATION.md` | 摘要前自动推进约定 |
| `docs/iclr2027/R4_CONFIRMATION_PROTOCOL.md` | R4/R5 独立确认协议 |
| `docs/iclr2027/R4_PUBLICATION_EVIDENCE.md` | R4 论文成立所需的证据 |
| `docs/iclr2027/R4_RELATED_WORK_BOUNDARIES.md` | R4 相关工作与贡献边界 |
| `docs/iclr2027/R4_SCALE_INFORMATION_NOTE.md` | 全缺失目标与尺度信息的对照解释 |
| `docs/iclr2027/R4_SCREENING_FINDINGS.md` | R4 开发筛查结果 |
| `docs/iclr2027/R4_SPRINT_TO_ABSTRACT.md` | R4：面向下游 MAE/MSE 的摘要前实验 |

### R5

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R5_ALIGNED_PORTFOLIO_PLAN.md` | 与输出组合一致的源教师评分 |
| `docs/iclr2027/R5_ALIGNED_PORTFOLIO_RESULTS.md` | 43 选项共享评分的开发结果 |
| `docs/iclr2027/R5_BLOCK_COMPOSER_PROGRESS.md` | R5：共享块级组合器开发 |
| `docs/iclr2027/R5_BUDGET_ORIGIN_EXTENSION.md` | 训练预算结论的额外历史起点检查 |
| `docs/iclr2027/R5_BUDGETED_STUDENT_PORTFOLIO.md` | 固定三候选预测组合的开发检查 |
| `docs/iclr2027/R5_COMPLETE_HISTORY_TEACHER_CHECK.md` | 完整历史预测教师的可行性检查 |
| `docs/iclr2027/R5_DATA_AND_RUNTIME.md` | 当前开发面板与运行设置 |
| `docs/iclr2027/R5_DIFFERENTIABLE_COMPOSITION_ASSESSMENT.md` | 可微填补组合：候选路线评估 |
| `docs/iclr2027/R5_FOLLOWUP_CONFIRMATION_PROTOCOL.md` | Frozen follow-up after matched portfolio development |
| `docs/iclr2027/R5_FOLLOWUP_MOTM_SUPPLEMENT.md` | MoTM follow-up comparator supplement |
| `docs/iclr2027/R5_FOLLOWUP_RESULTS.md` | 新验证结果与研究判断 |
| `docs/iclr2027/R5_FOLLOWUP_RUNTIME_AND_CONTROLS.md` | 新验证的执行和对照说明 |
| `docs/iclr2027/R5_FOLLOWUP_TEACHER_DIAGNOSTIC.md` | Post-evaluation teacher-transfer diagnostic |
| `docs/iclr2027/R5_FORECAST_PROJECTION_PLAN.md` | 分开计算已知偏移与学习教师残差投影 |
| `docs/iclr2027/R5_FORECAST_RESPONSE_TEACHER_PLAN.md` | 候选预测信息与完整历史教师监督 |
| `docs/iclr2027/R5_GAUSSIAN_INTEGRATION_PROTOCOL.md` | 条件高斯填补与预测组合开发对照 |
| `docs/iclr2027/R5_GEOMETRY_FIGURE_DESIGN.md` | 输出集合示意图 |
| `docs/iclr2027/R5_HORIZON_CONSENSUS_PROTOCOL.md` | 预测输出决策粒度对照 |
| `docs/iclr2027/R5_IMPUTER_BUDGET_STUDY.md` | 神经填补训练预算的开发对照 |
| `docs/iclr2027/R5_INPUT_SCOPE_INTERVENTIONS.md` | 目标与其他变量填补的交叉替换 |
| `docs/iclr2027/R5_MATCHED_PORTFOLIO_OBJECTIVES.md` | 同一配对分类器下的组合目标检查 |
| `docs/iclr2027/R5_MATCHED_PORTFOLIO_RESULTS.md` | 同学习器的组合目标结果 |
| `docs/iclr2027/R5_PAIRWISE_SELECTION_PROTOCOL.md` | 成本敏感两两选择与回归对照 |
| `docs/iclr2027/R5_PREFORECAST_PROJECTION_PLAN.md` | 预测前选择与可实现共识目标 |
| `docs/iclr2027/R5_RESEARCH_DECISION_20260913.md` | 2026-09-13：独立确认结果与论文路线判断 |
| `docs/iclr2027/R5_SELECTOR_LEARNING_CURVE_PROTOCOL.md` | 固定选择器的独立历史学习曲线 |
| `docs/iclr2027/R5_SHARED_FORECAST_GATE_PLAN.md` | Small shared forecast gate: matched source-development experiment |
| `docs/iclr2027/R5_SHARED_GATE_RESULTS.md` | 小型共享预测加权网络的源开发结果 |
| `docs/iclr2027/R5_SHARED_GATE_TRANSFER_DIAGNOSTIC.md` | Fixed shared-gate transfer diagnostic on used data |
| `docs/iclr2027/R5_STRUCTURED_STUDENT_PLAN.md` | 输入变化表示的有界改进试验 |
| `docs/iclr2027/R5_TRIPLE_OBJECTIVE_DIAGNOSTIC.md` | 单候选排序与三候选组合目标的诊断 |
| `docs/iclr2027/R5_VECTOR_IMPUTATION_NOTE.md` | 固定多输出预测器的单次填补限制 |

### R6

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R6_CONFIRMATION_PLAN.md` | Independent confirmation of the fixed shared forecast gate |
| `docs/iclr2027/R6_CONFIRMATION_RESULTS.md` | R6 独立确认结果与研究判断 |
| `docs/iclr2027/R6_CONFIRMATION_SOURCE_STATUS.md` | 后续确认的数据使用状态 |
| `docs/iclr2027/R6_COORDINATE_TEACHER_DIAGNOSTIC.md` | 逐预测位置教师参照检查 |
| `docs/iclr2027/R6_COORDINATE_TEACHER_RESULTS.md` | 逐位置教师参照的结果 |
| `docs/iclr2027/R6_DECISION_CLASS_DIAGNOSTIC.md` | 预测组合决策类的可达误差检查 |
| `docs/iclr2027/R6_DECISION_CLASS_RESULTS.md` | 区间固定与逐位置组合的可达空间 |
| `docs/iclr2027/R6_GEOMETRY_GATE_PLAN.md` | 候选身份与预测关系输入的匹配检验 |
| `docs/iclr2027/R6_GEOMETRY_GATE_RESULTS.md` | 候选关系输入的匹配结果 |
| `docs/iclr2027/R6_INTERVAL_GATE_PLAN.md` | 区间输入的匹配源端对照 |
| `docs/iclr2027/R6_INTERVAL_GATE_RESULTS.md` | 区间输入的匹配源端结果 |
| `docs/iclr2027/R6_LATENT_INTERFACE_PROBE_PLAN.md` | 冻结预测器内部表示的接口检查 |
| `docs/iclr2027/R6_LEGACY_NATIVE_GATE_PLAN.md` | 固定共享门控的原始缺失覆盖扩展 |
| `docs/iclr2027/R6_LEGACY_NATIVE_GATE_RESULTS.md` | 固定共享门控的九家族原始缺失扩展 |
| `docs/iclr2027/R6_LOCAL_CALIBRATION_DIAGNOSTIC.md` | R6 确认后的固定组合与前缀支持检查 |
| `docs/iclr2027/R6_METHOD_EXPLANATION_ZH.md` | 当前方法：面向下游预测的填补组合 |
| `docs/iclr2027/R6_NATIVE_REFERENCE_NUMERICAL_NOTE.md` | 训练扩展中的旧基线批次重放 |
| `docs/iclr2027/R6_NATIVE_SOURCE_TRANSFER_PLAN.md` | 原始缺失历史的跨家族训练扩展 |
| `docs/iclr2027/R6_NATIVE_SOURCE_TRANSFER_RESULTS.md` | 原始缺失训练扩展结果 |
| `docs/iclr2027/R6_NUMERICAL_REPLAY_NOTE.md` | R6 inference-layout correction |
| `docs/iclr2027/R6_ORIGIN_WEIGHTING_PLAN.md` | 源历史权重的匹配对照 |
| `docs/iclr2027/R6_ORIGIN_WEIGHTING_RESULTS.md` | 原始历史等权训练的结果 |
| `docs/iclr2027/R6_POSITIONAL_PORTFOLIO_PLAN.md` | 以中位数为起点的逐位置预测组合 |
| `docs/iclr2027/R6_POSITIONAL_PORTFOLIO_POSITIONING.md` | 逐位置组合与已有工作的关系 |
| `docs/iclr2027/R6_POSITIONAL_PORTFOLIO_RESULTS.md` | 逐位置预测组合结果 |
| `docs/iclr2027/R6_PREFIX_CALIBRATION_PLAN.md` | 目标前缀教师校准：固定设置的开发试验 |
| `docs/iclr2027/R6_PREFIX_CALIBRATION_POSITIONING.md` | 前缀校准与既有预测组合工作的关系 |
| `docs/iclr2027/R6_PREFIX_CALIBRATION_RESULTS.md` | 固定前缀教师校准的结果 |
| `docs/iclr2027/R6_PREFIX_LABEL_CONTROL_PLAN.md` | 历史教师与历史真实标签的匹配对照 |
| `docs/iclr2027/R6_PROBABILISTIC_OUTPUT_READINESS.md` | 概率输出的现有证据与下一项检查 |
| `docs/iclr2027/R6_READOUT_STRATEGY.md` | R6 result interpretation fixed before accuracy readout |
| `docs/iclr2027/R6_RESEARCH_DIRECTION_20260914_PM.md` | 9月14日下午的研究判断 |
| `docs/iclr2027/R6_REVIEW_20260914.md` | R6 工作稿的内部研究审查 |
| `docs/iclr2027/R6_SOURCE_LABEL_OBJECTIVE_PLAN.md` | 同一共享网络的源监督目标验证 |
| `docs/iclr2027/R6_SOURCE_LABEL_OBJECTIVE_RESULTS.md` | 同一网络的源监督目标比较 |
| `docs/iclr2027/R6_SOURCE_QUANTILE_AUDIT_PLAN.md` | 原源开发分位数完整性检查 |
| `docs/iclr2027/R6_SOURCE_QUANTILE_AUDIT_RESULTS.md` | 源分位数检查结果 |
| `docs/iclr2027/R6_SUPERVISED_CONTROL.md` | Matched source-future supervision control |
| `docs/iclr2027/R6_TEACHER_TRANSFER_DIAGNOSTIC.md` | R6 完整历史教师与迁移损失诊断 |
| `docs/iclr2027/R6_TEACHER_TRANSFER_RESULTS.md` | 完整历史教师与权重迁移诊断结果 |

### R7

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R7_LATENT_GENERALIZATION_PROTOCOL.md` | 内部表示的时间与家族泛化诊断 |
| `docs/iclr2027/R7_LATENT_GENERALIZATION_RESULTS.md` | 内部表示泛化诊断结果 |
| `docs/iclr2027/R7_LATENT_SOURCE_PROTOCOL.md` | 冻结预测器表示的源端准确度试验 |
| `docs/iclr2027/R7_LATENT_SOURCE_RESULTS.md` | 冻结预测器表示的源端结果 |

### R8

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R8_CALIBRATED_GATE_PROTOCOL.md` | 源时间校准与预测偏离约束试验 |
| `docs/iclr2027/R8_CALIBRATED_GATE_RESULTS.md` | 时间校准与偏离约束的结果 |

### R9

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R9_SOURCE_EXPANSION_PROTOCOL.md` | 不同源历史数量与优化步数的匹配试验 |
| `docs/iclr2027/R9_SOURCE_EXPANSION_RESULTS.md` | 扩充源训练历史的结果 |
| `docs/iclr2027/R9_TRANSFER_PROTOCOL.md` | 固定源模型的短迁移核验 |
| `docs/iclr2027/R9_TRANSFER_RESULTS.md` | 固定源模型的迁移核验结果 |

### R10

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R10_METRIC_OBJECTIVE_PROTOCOL_V002.md` | 固定组合数值精修：R10第二次执行 |
| `docs/iclr2027/R10_METRIC_OBJECTIVE_PROTOCOL.md` | 预测MAE与MSE训练目标的匹配比较 |
| `docs/iclr2027/R10_METRIC_OBJECTIVE_RESULTS.md` | MAE与联合预测目标的源端结果 |
| `docs/iclr2027/R10_SOLVER_PRECISION_NOTE.md` | R10固定求解器精度诊断 |
| `docs/iclr2027/R10_TRANSFER_PROTOCOL.md` | R10固定模型的短迁移评估 |
| `docs/iclr2027/R10_TRANSFER_RESULTS.md` | R10目标变化的迁移结果 |

### R11

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R11_TARGET_LOCAL_PROTOCOL.md` | Chronos目标级特征与共享特征的匹配比较 |
| `docs/iclr2027/R11_TARGET_LOCAL_RESULTS.md` | Chronos目标级特征的源端结果 |
| `docs/iclr2027/R11_TRANSFER_PROTOCOL_V002.md` | 共享对照的推断次数修正 |
| `docs/iclr2027/R11_TRANSFER_PROTOCOL.md` | 目标级选择的短迁移评估 |
| `docs/iclr2027/R11_TRANSFER_RESULTS.md` | 目标级选择的迁移结果 |

### R12

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R12_MOTM_POOL_PROTOCOL.md` | 含MoTM的候选池匹配试验 |
| `docs/iclr2027/R12_SOURCE_RESULTS.md` | 八候选池的源端结果 |
| `docs/iclr2027/R12_TRANSFER_PROTOCOL.md` | 同八候选池的短迁移比较 |
| `docs/iclr2027/R12_TRANSFER_RESULTS.md` | 同八候选池的迁移结果 |

### R13

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R13_REAL_POOL_PROTOCOL.md` | 真实缺失训练分布的同预算比较 |
| `docs/iclr2027/R13_REAL_POOL_RESULTS.md` | 真实缺失源训练分布的结果 |

### R14

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R14_CONDITIONAL_RISK_POSITION.md` | 条件风险诊断的研究定位 |
| `docs/iclr2027/R14_CONDITIONAL_RISK_PROTOCOL.md` | 已知条件未来分布的受控诊断 |
| `docs/iclr2027/R14_CONDITIONAL_RISK_RESULTS.md` | 条件风险诊断结果 |

### R15

| 文件 | 文档标题 |
|---|---|
| `docs/iclr2027/R15_AUDIT_REPLAY_NOTE.md` | R15核验的数组排列修正 |
| `docs/iclr2027/R15_DEPENDENCE_PROTOCOL.md` | 保持期望预测风险的未来相关性干预 |
