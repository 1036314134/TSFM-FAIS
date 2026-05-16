# 工作2：面向下游预测效用的填补策略选择

## 1. 题目备选

中文题目可以考虑：面向时序基础模型预测效用的缺失填补策略选择；FAIS：面向时序基础模型的预测感知填补器选择；从固定填补到场景化选择：面向 TSFM 的时序缺失修复策略推荐；面向下游预测的时序缺失填补选择；无统一最优填补器：面向 TSFM 的下游效用感知填补决策。

英文题目可以考虑：Forecast-Aware Imputer Selection for Time Series Foundation Models；FAIS: Forecast-Aware Imputer Selection for TSFM Forecasting；Downstream-Aware Imputation Selection for Time Series Foundation Models；Choosing the Right Imputer for Foundation Model Forecasting；Scenario-Aware Imputation Strategy Selection for Time Series Foundation Models。

## 2. 论文定位

工作2建议作为第二篇推进。它可以最大程度复用工作1的实验结果，技术风险低于直接提出新填补器，并且有明确独立性。它的核心问题是：给定数据结构、缺失模式和目标 TSFM，应该选择哪个填补策略才能获得较低的下游预测风险。由于 TimesFM、Chronos、Sundial 等模型的输入处理和预测输出形式存在差异，固定采用某一个填补方法很难成为稳健的部署策略。([Proceedings of Machine Learning Research][1])

## 3. 一句话背景与意义

不同数据域、缺失几何和目标 TSFM 下的最优填补器并不稳定，真实部署中固定使用单一填补策略容易产生不可控的下游预测风险。

## 4. 一句话方法亮点与方法概述

本文将填补策略选择形式化为场景条件下的最小 regret 决策问题，并利用数据结构特征、缺失几何特征、候选填补器代理扰动和目标 TSFM 信息预测各填补器的下游风险。

## 5. 核心研究问题

RQ1：是否存在跨模型、跨数据集、跨缺失率稳定最优的填补器。

RQ2：数据结构特征和缺失几何特征能否预测最佳填补器。

RQ3：候选填补器的代理结构扰动能否提升选择效果。

RQ4：selector 相比固定 linear 或固定平均最优方法，能否降低 downstream regret。

## 6. 任务形式化

定义场景为

[
s=(D,v,f,p,r,L,g,M),
]

其中 (D) 表示数据集或数据域，(v) 表示变量，(f) 表示采样频率，(p) 表示缺失模式，(r) 表示缺失率，(L) 表示块长，(g) 表示缺失几何特征，(M) 表示目标 TSFM。候选填补器集合可以定义为

[
\mathcal{I}={\text{mean},\text{forward},\text{backward},\text{linear},\text{seasonal},\text{Kalman},\text{native}}.
]

目标是选择

[
i^*(s)=\arg\min_{i\in\mathcal{I}}\Delta_{\text{forecast}}(i\mid s),
]

其中 (\Delta_{\text{forecast}}) 表示相对于 clean prediction 的下游损失增加。主评价指标应采用 regret：

[
\text{regret}(s)
================

## \Delta_{\text{forecast}}(\hat{i}\mid s)

\min_{i\in\mathcal{I}}\Delta_{\text{forecast}}(i\mid s).
]

这种定义比单纯分类准确率更符合实际使用，因为选择器即使没有选中 oracle best，只要选中的方法性能接近 oracle best，实际风险也较低。

## 7. 特征设计

数据结构特征包括趋势强度、趋势线性、季节强度、季节相关、残差自相关、谱熵、ADF 平稳性、主周期估计、序列长度和采样频率。缺失几何特征包括缺失率、最大块长、平均块长、块数量、最后一个缺失块到预测起点的距离、缺失是否靠近 context 尾部、缺失块占主周期比例。候选填补器代理特征包括 pseudo-mask reconstruction error、structure drift、smoothness change、low-frequency energy change、boundary discontinuity 和 observed-position distortion。模型特征包括 TSFM one-hot、context length、输出类型、是否支持 native missing，以及是否输出样本或分位数。

其中，候选填补器代理特征是工作2的关键。选择器不应只看原始数据和缺失模式，还应看每个候选填补器会怎样改变数据结构。这样可以把工作1中的机制变量转化为可用于决策的输入特征。

## 8. 方法路线

第一层是 rule-based selector，用于提供可解释基线。例如，强季节且块长接近主周期时优先考虑 seasonal 或 Kalman；序列平滑且缺失块较短时优先考虑 linear；目标模型对平滑输入更敏感时可以考虑 mean 或强平滑策略。

第二层是 classifier selector，将每个 scenario 的最佳填补器作为类别标签，使用 logistic regression、random forest 或 XGBoost 预测最优方法。该路线实现简单，但容易忽略不同错误选择之间的代价差异。

第三层是 ranker selector，建议作为主方法。它以 scenario features、imputer proxy features 和 model features 为输入，预测每个候选填补器的 forecast degradation，再选择预测风险最低的方法。可选模型包括 RandomForestRegressor、XGBoostRegressor、LightGBM ranker 和 ElasticNet。ElasticNet 可以作为解释性基线，树模型可以作为主实验方法。

## 9. 数据切分

工作2必须避免随机行切分，因为随机切分容易让相似数据集、相似缺失率和相同模型同时出现在训练集和测试集中，从而高估选择器能力。主文至少报告 leave-dataset-out，附录可以报告 leave-model-out 和 leave-ratio-out。leave-dataset-out 用于验证跨数据域泛化，leave-model-out 用于验证对新 TSFM 的迁移能力，leave-ratio-out 用于验证对新缺失强度的稳健性。

## 10. 实验设计与评价指标

主实验候选填补器可以先使用 mean、forward、backward、linear、seasonal、Kalman 和 native。后续可以把工作3的 SPImpute 作为扩展候选加入，但工作2主文不应依赖工作3。主评价指标为 regret、gain versus linear、top-2 hit、best-10% hit、低风险率和 selector stability。低风险率可以定义为选择结果的预测损失不超过 oracle best 某一阈值的比例，例如 10%。

## 11. 主图设计

主图可以包括 FAIS 框架图；linear default、average-best、rule selector、classifier selector、ranker selector 和 oracle 的 regret 对比图；leave-dataset-out 结果表；特征重要性图；固定 linear 失败但 selector 成功的 case study。

## 12. 与其他工作的边界

工作2不提出新填补器，也不研究多重填补。它的贡献是将已有填补器的使用从固定策略转化为场景化选择，并用 regret 度量选择结果的下游代价。它可以利用工作1的结构漂移指标，也可以在扩展实验中加入工作3的方法，但工作2本身应独立成立。

## 13. 最小可发表版本

最小版本可以包括 mean、forward、backward、linear 和 seasonal 五类候选填补器，使用数据结构、缺失几何、TSFM one-hot 和 imputer proxy drift 作为特征，方法上实现 rule selector 和 random forest ranker，切分采用 leave-dataset-out，指标报告 regret 和 gain versus linear。