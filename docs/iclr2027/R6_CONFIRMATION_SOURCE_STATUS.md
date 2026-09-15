# 后续确认的数据使用状态

2026-09-14。下表来自当前实际输入清单，只用于排除已经参与开发或评价的来源。每行的历史数按该清单内的origin_id去重，不能直接相加解释为独立样本数。频率、序列别名和原始来源重合还需另行核对。

| 已使用阶段 | 输入任务 | 清单内历史 | 家族标识 |
|---|---:|---:|---:|
| 原扩展源开发 | 7,812 | 217 | 15 |
| R5原始缺失确认 | 373 | 373 | 9 |
| R5后续确认 | 823 | 263 | 7 |
| R6确认 | 1,522 | 381 | 4 |

四份清单合计涉及32个不同的family_id。这是使用记录中的标识数量，不代表32个独立确认来源。原源开发中的217个历史包含165训练历史和52验证历史。R5后续确认与先前确认共享sg_weather、smart_manufacturing和water_quality_darwin家族，另有solar_alabama、uci_air_quality、uci_household_power与weather。R6另含appliances、beijing_multisite、bike_sharing和occupancy。

UCI Air Quality与Individual Household Electric Power Consumption已在 `R5_FOLLOWUP_CONFIRMATION_PROTOCOL.md` 及后续输入清单中登记，故不能再作为新的独立确认来源。官方Air Quality页面记载缺失标记为−200，Household Power页面记载分钟网格完整而部分测量缺失；这些信息有助于解析原始数据，但不改变它们已用于本项目评价的事实。[Air Quality官方记录](https://archive.ics.uci.edu/dataset/360/air+quality)，[Household Power官方记录](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)。

本次只读取来源、任务数量和既有使用记录，没有下载新数据、拟合模型或查看候选新来源上的预测误差。尚未选定或启动新的独立确认集合。后续准入仍需先核对原始来源、来源别名、时间网格、原始缺失语义、训练前缀及未来观测覆盖，再冻结方法与确认协议。

核对清单为 `artifacts/iclr27-r3/development-expanded-v001/episodes_manifest.json`、`artifacts/iclr27-r5/native-confirmation-v001/prepared/manifest.json`、`artifacts/iclr27-r5/followup-confirmation-v001/prepared/manifest.json` 和 `artifacts/iclr27-r6/confirmation-v001/prepared/manifest.json`。
