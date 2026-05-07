# TSFM-FAIS

## 题目方向

本仓库对应工作 3：**面向下游预测效用的填补策略选择**。

推荐题目：

- **Forecast-Aware Imputer Selection for Time Series Foundation Models**
- **No One-Size-Fits-All: Downstream-Aware Imputation Selection for TSFMs**
- **FAIS: Forecast-Aware Imputer Selection for Time Series Foundation Model Forecasting**
- **Choosing the Right Imputer for Foundation Model Forecasting**

中文题目备选：

- **面向时序基础模型预测效用的缺失填补策略选择**
- **没有统一最佳填补器：面向 TSFM 的下游效用感知填补选择**
- **FAIS：面向时序基础模型的预测感知填补器选择**
- **面向下游预测的时序缺失修复策略推荐**
- **从固定填补到场景化选择：面向 TSFM 的 imputer selection**

## 三句话总结

| 项目 | 内容 |
| --- | --- |
| 背景精华 | 固定使用某个填补方法不符合 TSFM 部署实际，因为最佳填补器会随数据结构、缺失形态和目标模型变化。 |
| 最突出贡献 | 提出 `FAIS`，把填补器选择从经验规则变成面向下游预测效用的 model-aware、data-aware 决策问题。 |
| 计算问题 | 上下文排序或最小 regret 选择，即预测各候选填补器在当前场景下的 forecast degradation 并选择最低风险方案。 |

## 论文定位

本工作是一篇选择/决策论文。它不提出新的填补器，而是解决：

> 给定数据结构、缺失模式和目标 TSFM，应该选择哪个填补策略才能获得最小下游预测退化？

动机来自已有发现：不同 TSFM、数据域和缺失条件下不存在统一最佳填补器。`linear` 通常稳健，但并不总是最优；`mean` 平均风险较高，但在 `visiontspp` 等模型上可能通过强平滑获得收益。因此，真实部署中更合理的做法不是固定使用某个填补器，而是根据场景进行选择。

本工作与其他并行工作的边界：

- 不做工作 1 的机制分析，只使用其结构特征和退化结果作为选择依据。
- 不做工作 2 的新填补器，只把 `SPImpute` 当成可选候选方法。
- 不做工作 4 的多重填补和不确定性传播，只选择一个或一类策略。

## 核心研究问题

1. `RQ1`：是否存在跨模型、跨数据集稳定最优的填补器？
2. `RQ2`：数据结构特征和缺失几何特征能否预测最佳填补器？
3. `RQ3`：候选填补器的代理结构扰动能否提升选择效果？
4. `RQ4`：selector 相比固定 `linear` 或固定最佳平均方法，能否降低 downstream regret？

## 方法名称

建议使用：

```text
FAIS: Forecast-Aware Imputer Selection
```

## 任务形式化

定义场景：

```text
scenario = (dataset, variate, frequency, missing_pattern, missing_ratio, block_length, target_TSFM)
```

候选填补器集合：

```text
I = {mean, forward, backward, linear, seasonal, Kalman, native}
```

可扩展：

```text
I = I + {SPImpute, multiple-imputation}
```

目标：

```text
select i* = argmin_i ForecastDegradation(i | scenario)
```

标签来自已有预测结果：

- 预测 MSE 最优填补器。
- 预测 sMAPE 最优填补器。
- 相对 clean 退化最小填补器。

## 特征设计

### 数据结构特征

- 趋势强度
- 趋势线性
- 季节强度
- 季节相关
- 残差自相关
- 谱熵
- ADF 平稳性
- 主周期估计
- 序列长度
- 采样频率

### 缺失几何特征

- 缺失率
- 最大块长
- 平均块长
- 块数量
- 最后一个缺失块到预测起点的距离
- 缺失是否靠近 context 尾部
- 缺失块占主周期比例

### 候选填补器代理特征

对每个 imputer 先计算：

- pseudo-mask reconstruction error
- structure drift
- smoothness change
- low-frequency energy change
- observed-position distortion

这部分是 `FAIS` 的关键。选择器不是只看数据，也看每个候选填补器会怎样改变数据。

### 模型特征

- TSFM one-hot
- context length
- 输出类型：点预测、分位数、采样
- 是否支持 native missing

## 方法路线

### Rule-based Selector

可解释规则：

```text
if target_model == visiontspp and sequence is noisy:
    prefer mean or strong smoothing
elif seasonality is strong and block_length >= 0.5P:
    prefer seasonal or structure-preserving imputation
else:
    prefer linear
```

作为 baseline 和解释工具。

### Classifier Selector

多分类：

```text
input: scenario features
label: best imputer
model: logistic regression / random forest / XGBoost
```

### Ranker Selector

主方法建议用 ranker：

```text
input: scenario features + imputer proxy features + model features
output: predicted degradation
select imputer with minimum predicted degradation
```

可用模型：

- `RandomForestRegressor`
- `XGBoostRegressor`
- `LightGBM ranker`
- `ElasticNet` 作为解释性 baseline

## 数据切分

避免随机行切分，必须做更严格切分：

1. `leave-dataset-out`
2. `leave-model-out`
3. `leave-ratio-out`

主文至少报告 `leave-dataset-out`。如果要证明泛化能力更强，可以额外报告 `leave-model-out`。

## 评价指标

不要只看分类准确率，主指标应当是 regret：

```text
regret = degradation(selected) - degradation(oracle_best)
```

其他指标：

- `gain_vs_linear`
- top-2 hit
- best-10% hit
- 低风险率：预测退化不超过 10% 的比例
- selector 稳定性

## 实验设计

### 候选填补器

最小候选集：

- `mean`
- `forward`
- `backward`
- `linear`
- `seasonal`

增强候选集：

- `Kalman`
- `native`
- `SPImpute`
- multiple-imputation summary strategy

为了并行投稿，主实验可以先用已有 imputer；`SPImpute` 和 multiple imputation 作为扩展候选，不作为本文成立的必要条件。

### TSFM 模型

建议至少包含：

- `chronos2`
- `timesfm2p5`
- `sundial`
- `visiontspp`

如果已有结果允许，可加入：

- `timesfm2p0`
- `kairos23m`
- `kairos50m`

### 缺失设置

主实验：

- `BM`
- `length50`
- `10% / 20% / 30%`

扩展实验：

- 尾部缺失
- 相对块长 `0.5P / 1P / 2P`

## 主图设计

1. `FAIS` 框架图。
2. `linear default / average-best / rule / classifier / ranker / oracle` 的 regret 对比。
3. leave-dataset-out 结果表。
4. 特征重要性图。
5. case study：固定 `linear` 失败但 selector 成功的场景。

## 最小可发表版本

最小版本：

- 候选：`mean / forward / backward / linear / seasonal`
- 特征：数据结构 + 缺失几何 + TSFM one-hot + imputer proxy drift
- 方法：rule + random forest ranker
- 切分：leave-dataset-out
- 指标：regret + gain_vs_linear

## 风险与备选

| 风险 | 备选 |
| --- | --- |
| selector 准确率不高 | 改用 regret 指标，只要推荐方法接近 oracle 即可 |
| leave-model-out 太难 | 主文报告 leave-dataset-out，leave-model-out 放附录 |
| 特征过多导致解释困难 | 做特征分组消融和特征重要性分析 |
| 固定 linear 已很强 | 强调 selector 降低尾部风险和极端退化 |

