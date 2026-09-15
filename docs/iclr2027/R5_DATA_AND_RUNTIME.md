# 当前开发面板与运行设置

下表由 `development-expanded-v001/episodes_manifest.json` 的实际接纳任务和 `accuracy-development-v002/standardizers.json` 整理。每个数据集只评价一条多变量序列。起点是不同的时间位置，不代表同一序列中的窗口统计独立；每个起点的 36 个掩码任务也不作为 36 个独立历史。

| 数据集 | 家族 | 变量数 | 周期（步） | 训练起点 | 验证起点 | 训练掩码任务 | 验证掩码任务 |
|---|---|---:|---:|---:|---:|---:|---:|
| azure2019_D_5T | azure2019 | 3 | 288 | 16 | 4 | 576 | 144 |
| Coastal_T_S_H | coastal_ts | 3 | 24 | 16 | 4 | 576 | 144 |
| current_velocity_H | current_velocity | 6 | 24 | 10 | 4 | 360 | 144 |
| electricity | electricity | 321 | 24 | 16 | 4 | 576 | 144 |
| ETTh1 | ett | 7 | 24 | 16 | 4 | 576 | 144 |
| EWELD_Load_15T | eweld_load | 10 | 96 | 16 | 4 | 576 | 144 |
| exchange_rate | exchange_rate | 8 | 7 | 15 | 4 | 540 | 144 |
| national_illness | national_illness | 7 | 52 | 2 | 2 | 72 | 72 |
| NE_China_Wind_H | ne_china_wind | 4 | 24 | 16 | 4 | 576 | 144 |
| OpenElectricity_NEM_5T | openelectricity_nem | 10 | 288 | 16 | 4 | 576 | 144 |
| Port_Activity_D | port_activity | 2 | 7 | 4 | 4 | 144 | 144 |
| Supply_Chain_Customer_D | supply_chain | 36 | 7 | 4 | 4 | 144 | 144 |
| traffic | traffic | 21 | 24 | 16 | 4 | 576 | 144 |
| Uncertainty_1M_M | uncertainty | 3 | 12 | 1 | 1 | 36 | 36 |
| Vehicle_Sales_M | vehicle | 10 | 12 | 1 | 1 | 36 | 36 |
| 合计 | 15 个家族 | — | — | 165 | 52 | 5940 | 1872 |

掩码机制为 random_point、independent_block、synchronous_block、staggered_correlated、value_dependent 和 mixed_outage；源任务使用 0.1、0.3、0.5 三种缺失率和 6101、6102 两个种子。上下文和预测跨度均为 96 步，目标是第 0、1 个变量。不同频率下 96 步表示的实际时长不同，不能统一描述为 96 小时。训练前缀按各序列位置定义，评分统计量只从已观测的前缀值计算。

实现为先对整条序列生成掩码，再截取历史窗口。随机种子同时包含数据集、序列、时间划分、机制、名义缺失率和种子编号，因此同一划分内的不同起点继承同一整序列实现；训练和验证使用不同实现。相关变量选择以及数值依赖分数的中心、尺度由训练前缀提供。名义缺失率是生成器的整序列设置，不能当作每个窗口的实际缺失比例。数值依赖机制在整序列的高分位置放置缺失块，属于离线受控污染，不宣称模拟了所有在线缺失过程。

训练数量分布不均：五个较小家族（national_illness、port_activity、supply_chain、uncertainty、vehicle）合计只有 12 个训练起点。在完整 15 家族等权的源目标中，它们合计占三分之一的家族权重；家族留出时按剩余家族重新归一化。该事实描述权重集中程度，不是统计有效样本量估计，也不单独证明它造成方法失败。

traffic 使用当前本地 21 变量快照，列为 0–19 和 OT，不将它等同于全量数百变量的标准交通面板。2026-09-13 核对迁移后的 CSV，SHA256 仍为 `4652f0f60ea8b1ecb6fe0656c8a11a8c0d31e9a8fb146bfc4a7a5eeb4de02eb4`，与原实验清单一致。

## 填补器预算配置

SAITS 的配置为两层、d_model=64、d_ffn=128、四个注意力头、d_k=d_v=16、dropout=0、MIT_weight=ORT_weight=1。TimeMixer++ 为一层、d_model=32、d_ffn=64、四个头、六个卷积核、top_k=3、一个下采样层、下采样窗口为 2、dropout=0，channel_independence 与 apply_nonstationary_norm 均为 false。两者 n_steps=96，n_features 随数据集维数设置，批量为 16，patience=None。预算比较改变训练轮数与掩码训练窗口数，不改变这些结构参数，也不声称已完成算法级超参数优化。

该结构依据已经保存的预算模型 constructor_params。评价序列数量与填补器拟合序列数量不同：数据集级神经填补器可使用原清单指定的多个训练序列前缀，源实验复用早期保存的拟合产物；例如 current_velocity_H 的拟合序列不等于当前评价的首条序列。具体成员由拟合清单恢复，不能将 165 个选择器源历史描述为全部神经填补训练数据。

## 冻结预测器版本

| 预测器 | 模型标识 | 实验快照版本 | 使用范围 |
|---|---|---|---|
| Chronos-2 | amazon/chronos-2 | 29ec3766d36d6f73f0696f85560a422f50e8498c | 源开发、原生确认、预算检查；联合多变量输入 |
| TimesFM 2.5 | google/timesfm-2.5-200m-pytorch | 1d952420fba87f3c6dee4f240de0f1a0fbc790e3 | 源开发、原生确认、预算检查；独立目标输入 |
| TiRex | NX-AI/TiRex | 63c740922493f5fbe60b277609ec62babfba2762 | 三数据集预算检查；独立目标输入 |

TiRex 固定纯 torch、batch size 1、关闭编译与 TF32；其扩展结果不是第三预测器的原生确认。各方法共享前缀标准化评分。预测器各自的原生缺失处理和声明回退仍需按协议区分。

## 数据路径迁移

源数据原位于相邻 SPImpute 仓库的 `data/Origin`，现在位于 `data/original`。历史实验清单保留生成时的绝对路径和摘要，不改写其来源记录。当前正在运行的配对对照只读取本仓库已经验证的预测与特征缓存，未重新读取迁移后的原始数据。

`configs/data/datasets.yaml` 的旧相对根目录尚需在活动队列结束后更新或提供明确迁移映射，再用于新的原始数据加载。traffic 的字节一致性已核对，其余原始来源在重新读取时仍应逐项核验。本文档不把迁移路径变化解释成数据内容变化。
