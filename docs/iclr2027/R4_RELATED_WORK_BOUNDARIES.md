# R4 相关工作与贡献边界

核查日期：2026-09-11。以下是针对当前方法路线的补充核查，不代表全面文献综述。

## 已确认的邻近工作

Chroma 的《Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting》已在其公开版本中报告模型组合。第 3.4 节采用时间序列交叉验证进行模型选择和预测加权；实验设置留出训练序列末尾 H 步作为验证窗口。附录 D 给出贪心集成选择。其主要创新还包括低成本构造专门化预测器组合。因此，使用完整 H 步历史回测再挑选候选，不能单独作为本项目的新算法贡献。[全文](https://arxiv.org/html/2510.06419v2)

《Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting》（ICML 2023）研究通过交叉验证学习预测组合，并允许权重随序列、预测位置和分位数变化。直接学习预测加权、按目标分别加权或加入简单收缩，都需要与这一类方法明确区分；其理论结论不能直接移用到存在依赖关系和非随机缺失反馈的当前问题。[论文索引](https://proceedings.mlr.press/v202/hasson23a.html)，[全文](https://proceedings.mlr.press/v202/hasson23a/hasson23a.pdf)

《Causal Analysis for Time Series Foundation Models》（2026-08）对 Chronos-2 与 TimesFM 2.5 的合成模式作干预分析。摘要涉及持续性、趋势、周期和状态切换等现象。因此，一般性地展示这两个模型存在模式偏差，也不足以构成本项目独特的发现。当前仅核查其索引与摘要，尚未复现。[原始来源](https://arxiv.org/abs/2608.24303)

另检索到 ZooCast 的模型—任务共同表示与排序方法；当前只获取搜索引擎收录的原始论文片段，OpenReview 全文访问受到验证页面限制，尚不据此确认其完整技术边界或发表状态。

《Selective Imputation for Multivariate Time Series Datasets with Missing Values》（TKDE 2023）已经研究选择部分缺失时间点填补，通过多任务高斯过程与多目标优化平衡不确定性和序列表示，并评价分类及异常检测。此处核查了作者机构的论文说明与 DOI，未阅读全文。[机构来源](https://nr.no/en/publication/2182126/)，[DOI](https://doi.org/10.1109/TKDE.2023.3240858)

《Impute With Confidence: A Framework for Uncertainty Aware Multivariate Time Series Imputation》（2025）进一步以不确定性进行选择性填补，摘要报告电子病历和死亡风险预测实验。因此，“只填有把握的位置”本身也是已有方向；如后续研究保留部分缺失状态，需要明确冻结预测目标和实际决策机制的增量。当前核查范围为作者、日期与摘要。[原始来源](https://arxiv.org/abs/2507.09353)

Chronos-2 维护者在官方仓库讨论中明确建议输入保留 NaN，并于 2026 年 4 月再次确认。原生缺失输入因而属于模型正常用法，不能在主要比较中省略，也不能将新增这一对照表述为本项目提出的新方法。[维护者说明](https://github.com/amazon-science/chronos-forecasting/discussions/443)

《Are Time-Indexed Foundation Models the Future of Time Series Imputation?》的 2026 年 2 月公开版本比较了 MoTM、TabPFN-TS 与经典及监督填补器。其主要指标评价缺失值重构，不能直接推断它们会改善本项目的冻结预测 MAE/MSE。该工作仍说明六候选开发池不足以代表所有现代填补方法；最终确认应检查较强预训练填补器加入后，方法收益是否保持。当前尚未在 FAIS 上运行这两种候选，也未改变筛查候选池。[正文与实现说明](https://arxiv.org/html/2511.05980v2)

9 月 12 日补充检索了冻结预测器的轻量适配工作。《Generalized Prompt Tuning: Adapting Frozen Univariate Time Series Foundation Models for Multivariate Healthcare Time Series》通过提示适配冻结的单变量模型，结合多变量信息，并评价医疗分类和流感预测。本次核查限于 PMLR 官方摘要，足以说明“预测器冻结，仅训练外部小模块”已有先例；它是否覆盖当前缺失块组合的具体机制仍需阅读全文判断。[论文索引](https://proceedings.mlr.press/v259/liu25a.html)

《Lightweight Online Adaption for Time Series Foundation Model Forecasts》（ICML 2025）的官方摘要描述 ELF-Forecaster 与 ELF-Weighter：前者学习当前数据分布，后者组合它与基础模型的预测。因此，后续若增加原生预测与填补后预测的加权模块，不能单独用“轻量在线加权”表述新颖性。本次未审读其完整训练与反馈协议，不能宣称它与 FAIS 信息条件完全一致。[论文索引](https://proceedings.mlr.press/v267/lee25ag.html)

MoTM 官方实现将分别在不同来源训练的隐式时间函数与当前观测上的岭回归结合。因此，多模型组合与根据观测调整组合系数也已有直接用于填补的先例。FAIS 的潜在差异须落实到冻结预测目标、信息条件及经过验证的收益，不能只依赖组合形式。本次核查了说明、推理入口和发布权重清单，尚未在 FAIS 任务上运行。[官方实现](https://github.com/EDF-Lab/MoTM/tree/b406660be4e9e147c3622b4e2f23c71789c26811)，[论文](https://arxiv.org/abs/2507.13207)

《What’s a good imputation to predict with missing values?》（NeurIPS 2021）已经区分重新学习预测函数与保留完整数据回归函数，分析条件均值填补的偏差、曲率与条件方差，并讨论标量连续回归函数下修正填补的存在性和连续性。一般的非线性目标失配、曲率效应或“填补准不等于预测准”不能单独作为 FAIS 的新理论贡献；冻结多步预测是否产生额外限制仍需独立论证。[原文第 4 节](https://proceedings.neurips.cc/paper/2021/file/5fe8fdc79ce292c39c5f209d734b7206-Paper.pdf)

检索到 ICLR 2022 的 supMIWAE 工作，公开论文片段已描述对缺失变量进行积分、通过多重样本平均预测，并允许向量输出。当前直接 PDF 访问仍受到浏览器验证限制，尚未审读完整版本；可以确认的是，普通“多重填补后组合预测”本身已有先例，不能因换成 TSFM 就直接认定新颖性。[公开论文入口](https://openreview.net/pdf?id=J7b4BCtDm4)

## 与同组并行项目的关系

2026-09-11 只读检查相邻 `E:/ZMY/Github/TSFM-RECA/README.md`：该项目当前已将“重构选择何时足够”和“历史预测校准的收益与预算”作为两个主要研究问题，并已有可观测历史校准实验。FAIS 的普通近期回测比较与这一范围明显接近，不能仅更换数据、模型或评价指标作为另一篇论文的核心差异。

FAIS 继续优先验证可部署的预测收益选择与实际填补组合，要求相对固定填补、直接缺失处理、预测集成和同历史校准有明确增量。若最终采用结论型主线，结论必须回答 RECA 尚未覆盖且由本项目独立证据支持的问题，例如联合预测器中的填补交互机制及其操作性后果。当前并未证明这些差异成立，也未修改相邻项目的任何文件或任务。

进一步只读核查 `E:/ZMY/Github/TSFM-SPImpute/README.md` 与 `Imputation/spimpute_query.py`：SPImpute-Q 已使用少量历史预测查询重排其内部候选，并包含历史掩码平移、候选去重和切换条件。FAIS 不能将这一类普通历史重排再次作为独立核心贡献。候选的可微组合路线及其尚未解决的新颖性问题记录于 `R5_DIFFERENTIABLE_COMPOSITION_ASSESSMENT.md`。

9 月 12 日早间的更新核查显示，RECA 已将主线调整为填补误差经过归一化、输入表示和推理规则后的传播机制，并明确不开发填补算法或选择器。其 README 将预测收益估计、历史反馈与候选选择划入 FAIS 的研究范围。此前对两项工作范围接近的记录保留为历史判断；当前应按最新边界组织稿件。FAIS 仍需相对公开的选择、集成和门控研究证明自己的增量，不能仅凭同组分工认定新颖性。

## 当前研究应证明什么

近期回测、固定预测加权和模型选择作为现有比较方法进入实验。可能具有实质价值的研究对象包括：真实可见的历史反馈不足或偏置时，如何判断是否值得改变填补策略；冻结预测器的独立/联合结构如何影响填补组合的实际效果；预测器默认缺失处理、训练前缀信息和更长历史是否解释了表面上的填补收益；在相同信息与查询预算下，是否能形成稳定的准确度改进。

若数据支持预测组合优于填补上下文组合，需要解释其与冻结预测器非线性之间的关系，并验证这一关系是否具有跨数据、跨预测器的一致性。若长上下文或合理的标准化已经解释主要收益，则应据实调整方法主线，不能将这些基础条件带来的效果归因于复杂选择器。

现有的近期反馈实现是用于检验这些问题的实验工具。论文主张需要由后续真实模型结果、控制实验及独立确认共同支持，目前不预设它已经构成足够的新方法。

9 月 12 日晚重新核验 Le Morvan 等的 NeurIPS 2021 原文第 4 节。条件均值填补与预测不匹配、曲率乘条件方差的误差项，以及标量回归下修正填补的存在性与连续性均已讨论。当前多输出可达集合方向需将“最优修正填补后仍无法达到向量条件均值”与上述工作区分；其数学表述简单，新颖性仍未核定。推导和适用条件见 `R5_VECTOR_IMPUTATION_NOTE.md`。[原文](https://proceedings.neurips.cc/paper/2021/file/5fe8fdc79ce292c39c5f209d734b7206-Paper.pdf)

supMIWAE 的 ICLR 2022 作者机构摘要及原文索引第 4 节已核验。该工作以重要性采样近似多次填补后的预测平均，实验固定预训练生成模型并更新判别模型；保留判别网络架构不能直接理解为固定其参数。全文入口受浏览器验证或 403 限制，当前不能标记为完整通读。[机构页面](https://orbit.dtu.dk/en/publications/how-to-deal-with-missing-data-in-supervised-deep-learning-2/)，[原文](https://openreview.net/pdf?id=J7b4BCtDm4)

进一步阅读 ICML 2023《Probabilistic Imputation for Time-series Classification with Missing Data》第 3 节，该工作扩展 supMIWAE 至具有 MNAR 机制的时序分类，联合目标包含分类、数据生成和缺失机制，并用额外观测遮蔽限制无意义填补。因而“给时序概率填补加入下游目标”也已有直接先例。[原文](https://proceedings.mlr.press/v202/kim23m/kim23m.pdf)

《Impute With Confidence》2025 预印本的引言与方法部分已阅读：使用 Monte Carlo dropout 估计填补不确定性，并在验证集选定阈值后进行选择性填补，评价包括医疗数据下游分类。因此不能将“不确定时保留缺失”“不确定性感知选择”本身当作 FAIS 新颖性。[原文](https://arxiv.org/html/2507.09353)

9 月 13 日阅读 Merlin 第 3 节：它用完整历史训练教师，再以表示与预测结果蒸馏、不同缺失率视图的对比学习训练预测学生。因而“通过教师增强缺失时序预测”已有直接先例。当前拟议学生输出填补选择，后续预测器保持冻结，教师来自同一不完整输入的候选预测；这些区别需要进一步证据支撑，不能仅用蒸馏名称作为贡献。[Merlin 原文](https://arxiv.org/html/2506.12459v1)

《Regularized Ensemble Forecasting for Learning Weights from Historical and Current Forecasts》的 2026 年 8 月版本摘要已核验，其组合权重同时利用当前预测和历史表现。因此普通的历史误差与当前预测共同加权也不是独立新颖性。当前仅阅读摘要，不对其实现或完整实验作进一步判断。[原文](https://arxiv.org/abs/2602.11379v2)

Auto-TSF 的论文标题、作者及 ICDE 2025 发表信息已在官方会议页面核验，全文尚未取得并通读。因此便宜代理模型与预测算法选择的关系仍是需要补查的近邻工作，当前不宣称已排除重合。[官方会议记录](https://ieee-icde.org/2025/research-papers/)

进一步核验 Goswami 等的 ICLR 2023《Unsupervised Model Selection for Time-series Anomaly Detection》，并阅读第 3 节及附录 A.2。其代理指标包括预测误差、合成异常表现和模型中心性；中心性按异常分数排序之间的 Kendall 距离计算近邻一致性。论文也明确指出，表现不好的模型可能聚集在一起，且部分排序汇总依赖多数排序可靠的前提。因此“一致性或共识可以提供无标签模型选择依据”已有直接研究，不能独立作为本工作的创新。其目标为异常检测模型排序，当前 FAIS 的待验证问题是固定预测器下的输入选择与预测前学生，两者的具体增量仍需实际证据。[原文](https://arxiv.org/html/2210.01078)

9 月 14 日新增近邻工作：Utama 等于 2026 年 6 月发表的 deterministic duel-based imputation 已将预测相关评分、两两比较、前三候选和中位数等聚合方式用于填补。已阅读出版社原文摘要及第 3.1–3.3 节；其聚合对象为填补值，之后训练循环预测模型。因而“配对比较后聚合前三”也不能作为独立创新。该文的选择评分具体时间边界仍需实现层面的核对，不能据当前阅读认定其存在信息泄露。FAIS 当前需要证实的是冻结预测器下实际返回预测组合的监督与迁移效果。[出版社原文](https://link.springer.com/article/10.1007/s44564-026-00001-6)

Cao 等的《Conversational Time Series Foundation Models》已核验 arXiv 摘要及作者信息，使用经微调的语言模型协调预测模型集成。当前未通读其完整方法，也未运行其代码；该记录只支持一般的集成协调已有先例，不能支持对其具体信息条件或成本作判断。[原文](https://arxiv.org/abs/2512.16022)

小型加权预测门控所用的成员误差与集成误差之差，已在 Krogh 与 Vedelsby 的 NIPS 1994 论文第 2 节中给出；已阅读原始 PDF 的该节及摘要。该节也明确提到可推广到多个输出。因此加权分散度恒等式及其向量推广均不能作为新定理，而且该恒等式不直接适用于三候选中位数。当前新实验仅检验已有恒等式所区分的训练目标在固定输入、模型容量和源监督下的效果。[原文](https://proceedings.neurips.cc/paper_files/paper/1994/file/b8c37e33defde51cf91e1e03e51657da-Paper.pdf)
