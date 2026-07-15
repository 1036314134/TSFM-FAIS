# TSFM-FAIS：面向下游 TSFM 预测的块级多算法缺失填补

TSFM-FAIS 实现 B-FAIS（Block-wise Forecast-Aware Imputer Selection）：给定一段带缺失的多变量时间序列、目标预测模型和计算预算，系统把缺失区域拆成变量级原子块，从可扩展候选池中为每个块分别选择填补算法，再将各候选的局部结果组装为完整上下文。一个样本中的不同缺失块可以使用不同算法。选择目标由下游 TSFM 的预测损失定义，填补误差只作为历史伪缺失证据之一。

当前版本为 `0.1.0`。仓库已经包含严格配置与公共数据契约、CSV/Arrow 多变量加载、完整性审计、六类缺失注入、20 个候选适配器、块提取与关系图、LightGBM 路由模型、确定性搜索、TSFM 教师标签、五个预测适配器、CLI、单元测试和离线合成 smoke 测试。截至 2026-07-14，当前工作区已使用本地数据和 checkpoint 完成首轮 `main-v1` 主实验：32 个多变量数据版本全部通过审计，五个 TSFM 的教师标签、19 个留一数据族路由折、两种预测模式的填补和最终预测评估均已完成。首轮结果不支持 B-FAIS 整体优于 LOCF；结果与限制见“主实验结果”一节。数据、模型和 `artifacts/` 仍不纳入版本控制。

## 问题定义

输入为单条多变量轨迹 `X ∈ R^{T×D}`、布尔观测掩码 `M ∈ {0,1}^{T×D}`、目标 `ForecastSpec` 和 `BudgetSpec`。`M[t,d]=True` 表示原观测值，缺失位置在 `SeriesBatch` 内统一规范为 `NaN`。系统对每个变量提取极大连续缺失区间 `b=[start,end)`；同期多个变量缺失时仍保留为多个原子块，以允许不同变量使用不同候选。

对缺失块集合 `B`、候选集合 `C` 和分配 `a_b∈C`，路由器最小化下面的结构化风险：

```text
E(a) = Σ_b R₁(b, a_b)
     + β Σ_(b,c)∈Edges ρ(b, c, a_b, a_c)
     + λ Σ_i 1[i 被启用] cost(i)
```

`R₁` 是结合伪缺失证据的单块预测风险，`ρ` 是相关块之间的交互风险，最后一项约束启用候选的额外成本。候选数、同时启用数、设备、运行时间和内存由 `BudgetSpec` 表达；当前推理代码执行候选数、同时启用数和设备约束，真实缺失与伪缺失两次候选调用共享累计耗时，在达到上限后跳过剩余候选。内存限制依据单次候选结束前后的进程 RSS 增量作事后检查。运行中的单个第三方模型不会被强制中断。

主要调用接口如下：

```python
from tsfm_fais.contracts import BudgetSpec, ForecastSpec
from tsfm_fais.pipeline import BlockwiseFAIS

pipeline = BlockwiseFAIS.load(
    config="configs/smoke.yaml",
    router_artifact="artifacts/example/router",  # 可省略；省略后使用确定性启发式风险
)

result = pipeline.impute(
    item=time_series_item,                         # TimeSeriesItem: [T, D]
    observed_mask=observed_mask,                  # True 表示原观测值
    forecast_spec=ForecastSpec(
        model_id="timesfm2p5",
        mode="independent_univariate",
        horizon=96,
        target_indices=(0,),                      # 预测可为单目标
    ),
    budget=BudgetSpec(max_candidates=6, max_active_candidates=3),
)
```

由分阶段 CLI 训练的路由器会在元数据中记录对应的 `fit-imputers` 产物目录。`BlockwiseFAIS.load` 据此按 `time_series_item.metadata["dataset_id"]` 延迟加载该数据集的冻结候选、训练中位数和相关矩阵，因此上述两参数调用可以使用完整的已训练候选池。对于外部或旧路由产物，也可以向 `load` 显式传入 `imputer_artifact_root=...`；如果路由器需要数据集级候选而产物未提供，调用会明确报错，不会静默缩减为无拟合候选。

填补阶段始终接收全部 `D` 个变量。预测阶段可以只预测一个目标，也可以预测多个目标；预测器是否支持原生多变量输入与填补是否为多变量任务相互独立。

## B-FAIS 流程

```mermaid
flowchart LR
    A["完整多变量源数据审计"] --> B["确定滚动预测起点"]
    B --> C["先截断到历史上下文"]
    C --> D["读取真实缺失或注入缺失"]
    D --> E["提取原子缺失块"]
    E --> F["建立块关系图"]
    F --> G["R0 风险与预算预筛"]
    G --> H["运行短名单候选"]
    H --> I["历史伪缺失证据"]
    I --> J["R1 单块风险与块对风险"]
    J --> K["确定性 Beam Search"]
    K --> L["块级结果组装"]
    L --> M["联合多变量或独立单变量 TSFM"]
    M --> N["预测损失与教师标签"]
```

块关系图包含同变量相邻边、跨变量时间重叠边，以及高相关变量的近邻边。推理时优先使用训练折保存的相关矩阵；未提供该统计时，使用当前历史上下文的稳健补齐估计并在路由记录中标明来源。特征覆盖块长度与位置、边界可用性、同期缺失比例、周期信息、变量相关性、候选能力与成本、目标 TSFM 能力，以及候选在历史伪缺失位置上的重构误差和结构变化。一次 episode 最多放置 8 个跨变量时间区间也不重叠的伪缺失块，每个入围候选只需增加一次填补调用。

两阶段路由均按“一个块的一组候选构成一个 ranking group”组织训练数据。`R0` 和 `R1` 使用 LightGBM LambdaMART，块对模型使用 LightGBM Huber 回归。默认预筛强制保留 `locf` 与 `linear_interp`，再按风险覆盖和成本加入候选，短名单上限为 6。搜索默认 beam width 为 32，并采用稳定排序处理并列；穷举求解器只用于小规模正确性测试。

教师标签通过反事实上下文生成。单块标签只替换 anchor 中一个块，块对标签同时替换两个相连块，并计算二阶交互项。默认预测损失是目标列宏平均 MASE；预测目标可以是一列或多列。教师模块只使用预测起点前的填补上下文和预测起点后的评估真值，评估真值不会进入填补器或路由特征。

## 20 个登记候选（主实验 19 个可执行）

所有候选接收统一的 `[N,L,D]` `SeriesBatch`。逐变量候选会遍历 `D` 个变量后重新组装完整张量；联合候选直接使用多变量结构。`fit` 只在训练折运行，`impute` 使用冻结 artifact。可选依赖采用延迟导入，缺少依赖只会将对应候选标记为 `UNAVAILABLE`。

| ID | 方法族 | 多变量处理方式 | 训练 artifact | 尾部块 | 依赖/说明 | 成本级别 |
|---|---|---|---|---|---|---:|
| `locf` | 持续性 | 逐变量 | 无 | 支持 | NumPy；前向保持并安全处理序列开头 | 1 |
| `linear_interp` | 线性插值 | 逐变量 | 无 | 不原生支持 | NumPy；仅内部块具有双侧边界 | 1 |
| `seasonal_lag` | 季节滞后 | 逐变量 | 无 | 支持 | 需要周期；使用历史同相位稳健统计 | 1 |
| `kalman_local_trend` | 状态空间 | 逐变量 | 无 | 支持 | 局部趋势平滑 | 2 |
| `kalman_ar` | 状态空间 | 逐变量 | 无 | 支持 | AR 状态建模 | 2 |
| `stl_kalman` | 分解与状态空间 | 逐变量 | 无 | 支持 | Statsmodels；需要周期 | 2 |
| `gp_rbf` | 高斯过程 | 逐变量 | 无 | 支持 | RBF 核；限制最大训练点数 | 3 |
| `knn_multivariate` | 近邻 | 联合多变量 | 数据集级 | 支持 | scikit-learn KNNImputer | 2 |
| `mice` | 链式回归 | 联合多变量 | 数据集级 | 支持 | scikit-learn IterativeImputer；随机种子固定 | 2 |
| `missforest` | 随机森林 | 联合多变量 | 数据集级 | 支持 | RandomForestRegressor 逐变量迭代 | 3 |
| `softimpute` | 低秩矩阵 | 联合多变量 | 数据集级 | 支持 | SciPy SVD 与奇异值收缩 | 2 |
| `trmf` | 时序低秩 | 联合多变量 | 数据集级 | 支持 | 已注册；PyPOTS 1.5 的实现为转导式接口，当前协议下标记为不可执行 | 3 |
| `brits` | 双向循环网络 | 联合多变量 | 数据集级 | 不原生支持 | `deep-imputers` 可选依赖 | 3 |
| `gpvae` | 概率潜变量 | 联合多变量 | 数据集级 | 支持 | `deep-imputers` 可选依赖 | 4 |
| `saits` | 自注意力 | 联合多变量 | 数据集级 | 支持 | `deep-imputers`；真实 `[N,L,D]` 联合训练 | 3 |
| `csdi` | 条件扩散 | 联合多变量 | 数据集级 | 支持 | `deep-imputers`；样本中位数与样本方差 | 5 |
| `imputeformer` | 时空注意力 | 联合多变量 | 数据集级 | 支持 | `deep-imputers` 可选依赖 | 4 |
| `helix` | 跨维度建模 | 联合多变量 | 数据集级 | 支持 | `deep-imputers` 可选依赖 | 4 |
| `timemixerpp` | 多尺度混合 | 联合多变量 | 数据集级 | 支持 | `deep-imputers`；适配 `TimeMixerPP` | 4 |
| `totem` | 时间序列 token | 联合多变量 | 数据集级 | 支持 | `deep-imputers` 可选依赖 | 4 |

公共组装器在候选返回后逐元素恢复所有原观测值。候选异常或原生输出含非有限值时，runner 会生成形状安全的有限张量，同时把对应缺失位置的 `native_valid_mask` 设为 `False`；路由器据此排除失败的“块—候选”组合，不会把安全补充值当作算法成功。`configs/router/block_fais.yaml` 声明内部块与尾部块的回退优先级；若某个块没有任何原生有效候选，推理 facade 优先使用显式传入的训练中位数，否则使用当前历史上下文中位数，并记录该块与回退来源。

### 增加新的填补算法

路由代码只依赖 `ImputerSpec` 与 `ImputerProtocol`，因此候选池不限于当前 20 个方法。新增候选需要实现 `fit`/`impute`，再注册一条带稳定 ID 的工厂记录；无需修改块提取、特征、搜索或组装器。

```python
from tsfm_fais.contracts import ImputerSpec
from tsfm_fais.imputers import IMPUTER_SPECS, ImputerRegistry

registry = ImputerRegistry(IMPUTER_SPECS)
registry.register(
    ImputerSpec(
        imputer_id="my_imputer",
        family="custom",
        mode="joint_multivariate",
        factory="my_package.my_module:MyImputer",
        fit_scope="dataset",
        supports_tail=True,
        stochastic=False,
        device="cpu",
        cost_tier=2,
        dependencies=("my_package",),
    )
)
```

适配器应返回 `CandidateResult`，其中包括完整候选张量、原生有效掩码、可选不确定性、耗时、内存和失败信息。可继承 `BaseImputer` 并只实现 `_fit` 与 `_impute_native`，公共代码会完成观测值恢复、有限值安全补全和状态判定。

## TSFM 预测适配

`ForecastRunner` 将两种模型能力统一到相同输出契约。

| 模式 | 输入处理 | 适用场景 | 统一输出 |
|---|---|---|---|
| `joint_multivariate` | 原样传入 `[N,L,D]` | 模型原生接收多变量上下文 | 点预测 `[N,H,K]`，分位数 `[N,H,K,Q]`，样本 `[N,S,H,K]` |
| `independent_univariate` | 将目标列展开为 `[N×K,L]`，批量预测后重组 | 模型只接收单变量序列 | 同上 |

默认登记的预测器如下。模型依赖和权重均不会在导入包时加载，只有显式构建并调用相应适配器时才需要安装 extra 或准备权重。

| ID | 模式 | 模型标识 | 原生输出 | 安装 extra |
|---|---|---|---|---|
| `chronos2` | 联合多变量 | `amazon/chronos-2` | 分位数 | `forecast-chronos` |
| `timesfm2p5` | 独立单变量 | `google/timesfm-2.5-200m-pytorch` | 分位数 | `forecast-timesfm` |
| `chronosbolt` | 独立单变量 | `amazon/chronos-bolt-base` | 分位数 | `forecast-chronos` |
| `sundial` | 独立单变量 | `thuml/sundial-base-128m` | 样本 | `forecast-sundial` |
| `tirex` | 独立单变量 | `NX-AI/TiRex` | 分位数 | `forecast-tirex` |

联合预测和独立单变量预测都发生在多变量填补之后。若只评估一个下游目标，可设置 `target_indices=(d,)`；填补器仍可使用所有变量。适配器不会在内部执行隐式线性填补或 last-value 补全，传入 TSFM 的上下文必须已经是有限值。其他预测模型可通过 `ForecastAdapterSpec + factory` 接入注册表。

## 数据准入与 32 个本地版本

实验数据必须来自原始完整的多变量序列。准入审计要求 `D≥2`，数值中无 `NaN`、null、`Inf` 或 manifest 声明的哨兵，变量名唯一，无常量列；显式时间轴必须可解析、严格递增、无重复并与声明频率一致。只有 `start+freq` 的 Arrow 数据必须在 manifest 中显式设置 `allow_implicit_regular_time: true`。CSV loader 显式识别时间列、可选 item 列和全部数值变量；Arrow IPC loader 按行返回独立 `TimeSeriesItem`，不会跨 item 拼接，也不会把多变量数据投影为单目标。

`configs/data/datasets.yaml` 当前登记 32 个本地多变量版本，共 8 个 CSV 和 24 个 Arrow IPC。文件保存在相邻仓库的数据目录中，不复制进本仓库；因此无需为了运行实验移动数据，只需保证 manifest 中的相对路径可解析。`data_root` 可按本机目录布局修改。`main-v1` 的审计结果为 32/32 通过，全部版本均进入主实验。

| 格式 | 数量 | 数据集 ID |
|---|---:|---|
| CSV | 8 | `electricity`, `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `national_illness`, `traffic` |
| Arrow | 8 | `azure2019_D_5T`, `azure2019_I_5T`, `azure2019_U_5T`, `Coastal_T_S_15T`, `Coastal_T_S_20T`, `Coastal_T_S_H`, `current_velocity_5T`, `current_velocity_15T` |
| Arrow | 8 | `current_velocity_20T`, `current_velocity_H`, `EWELD_Load_15T`, `Housing_Inventory_M`, `Job_Claims_W`, `JOLTS_M`, `NE_China_Wind_H`, `OpenElectricity_NEM_5T` |
| Arrow | 8 | `Port_Activity_D`, `Port_Activity_W`, `Supply_Chain_Customer_D`, `Supply_Chain_Location_D`, `Uncertainty_1M_M`, `US_Labor_M`, `Vehicle_Sales_M`, `Vehicle_Supply_M` |

每条记录带 `family_id`。32 个版本归并为 19 个数据族；`family_folds()` 以 family 为分组单位生成 leave-family-out 切分，避免 ETT、Azure、Coastal、Current Velocity、Port Activity、Supply Chain 和 Vehicle 等同源或不同频率版本分散到训练集与评估集；预测模型侧另有 `leave_model_out_folds()`。单变量版本和源文件本身含缺失的版本不进入该 manifest。后续更换或增加数据时仍须重新运行 `data audit`，不能沿用本次审计结论。

六种可复现缺失机制为：随机点、独立块、同步块、相关变量错位块、数值依赖块和尾部混合块。实验配置将缺失率限制在 `(0,0.5]`，与首版评估网格一致。块机制按变量规模扩展可放置块数，高维序列不会在配额不足时用全局随机点补齐；如果无法保持声明的结构并达到目标缺失量，注入器会明确报错。随机种子由数据集、item、预测起点、缺失配置和重复编号稳定派生。

## 仓库结构

```text
TSFM-FAIS/
├─ pyproject.toml
├─ environment.yml
├─ configs/
│  ├─ data/datasets.yaml
│  ├─ imputers/pool.yaml
│  ├─ forecasters/pool.yaml
│  ├─ router/block_fais.yaml
│  ├─ smoke.yaml
│  ├─ main.yaml
│  └─ main_eval.yaml
├─ src/tsfm_fais/
│  ├─ contracts.py
│  ├─ config.py
│  ├─ registry_configs.py
│  ├─ artifacts.py / stages.py
│  ├─ evaluation.py / main_results.py
│  ├─ label_artifacts.py / label_resume.py
│  ├─ stage_execution.py   # 仅由 run --execute 延迟导入
│  ├─ data/                 # loader、审计、缺失 episode
│  ├─ imputers/             # 20 个候选、注册表、runner
│  ├─ forecasting/          # TSFM 注册表、适配器、输出规范化
│  ├─ routing/              # 缺失块、关系图、特征、教师、模型、搜索
│  ├─ pipeline.py
│  ├─ cli.py
│  └─ __main__.py
└─ tests/
   ├─ unit/
   └─ integration/
```

## 安装

项目支持 Python `>=3.10,<3.12`。核心依赖包含 NumPy、Pandas、SciPy、scikit-learn、Statsmodels、PyArrow、LightGBM、Pydantic、PyYAML、joblib、psutil 和 tqdm。

使用 conda 创建开发环境：

```powershell
conda env create -f environment.yml
conda activate tsfm-fais
```

或在已有 Python 3.10/3.11 环境中安装：

```powershell
python -m pip install -e ".[dev]"
```

按需安装可选候选或预测器。项目有意不提供一次安装所有模型的 extra，以减少依赖冲突和无关下载。

```powershell
python -m pip install -e ".[deep-imputers]"
python -m pip install -e ".[forecast-chronos]"
python -m pip install -e ".[forecast-timesfm]"
python -m pip install -e ".[forecast-sundial]"
python -m pip install -e ".[forecast-tirex]"
```

安装 extra 只安装 Python 依赖，不代表模型权重已经下载。真实运行前应根据各模型许可、硬件和缓存策略单独准备权重。

## 配置

配置使用 Pydantic 严格校验，未知键会直接报错。注册表路径相对于主配置文件解析，数据路径相对于数据 manifest 解析。

```yaml
schema_version: 1
seed: 20260710
registries:
  data_manifest: data/datasets.yaml
  imputer_registry: imputers/pool.yaml
  forecaster_registry: forecasters/pool.yaml
  router_config: router/block_fais.yaml
experiment:
  split: rolling_origin
  context_length: 48
  horizon: 8
  target_indices: all
  missing_mechanisms: [independent_block, tail_mixed]
  missing_rates: [0.2]
  seeds: [20260710]
runtime:
  output_root: ../artifacts
  device: cpu
  fail_fast: true
```

上面的 YAML 是最小配置示例。`main-v1` 的训练配置为 `configs/main.yaml`：训练种子 20260710，每个数据版本最多 12 个教师 episode。最终评估配置为 `configs/main_eval.yaml`：种子 `{22,33,44}`，每个数据版本 24 个 episode，上下文 48、horizon 8、目标列 `[0,1]`，覆盖六种机制和四档缺失率；CSDI 使用 5 个填补样本，每个 TSFM 使用 20 个预测样本，并保存全部候选输出供配对评估。

候选元数据、预测器能力和路由参数分别位于 `configs/imputers/pool.yaml`、`configs/forecasters/pool.yaml` 和 `configs/router/block_fais.yaml`。代码注册表是运行时实现来源，YAML 是实验配置与审计清单；配置校验会逐项比较 ID 与关键能力字段，新增 ID 时应同步两者并通过配置测试。路由配置用 `beta` 和 `cost_weight` 指定本次搜索实际采用的权重，并要求它们属于相应候选网格；选定值、网格和训练标签尺度会写入路由产物，推理时从产物恢复。LambdaMART 输出先在每个缺失块内规范为稳定风险尺度，随后再与 Huber 块对交互项组合。

## CLI 与产物

安装后可使用 `fais`，也可使用 `python -m tsfm_fais`：

```powershell
python -m tsfm_fais config validate --config configs/smoke.yaml
python -m tsfm_fais data audit --manifest configs/data/datasets.yaml --output artifacts/data_audit.json
python -m tsfm_fais imputers list
python -m tsfm_fais forecasters list

python -m tsfm_fais run --config configs/smoke.yaml --stage fit-imputers --audit-artifact artifacts/data_audit.json
python -m tsfm_fais run --config configs/smoke.yaml --stage labels --audit-artifact artifacts/data_audit.json --imputer-artifacts artifacts/imputers --forecaster-id chronos2 --forecaster-artifact checkpoints/chronos2
python -m tsfm_fais run --config configs/smoke.yaml --stage labels --audit-artifact artifacts/data_audit.json --imputer-artifacts artifacts/imputers --forecaster-id chronos2,timesfm2p5 --forecaster-artifact checkpoints/forecasters.json
python -m tsfm_fais run --config configs/smoke.yaml --stage train-router --labels-artifact artifacts/teacher_labels.jsonl
python -m tsfm_fais run --config configs/smoke.yaml --stage impute --audit-artifact artifacts/data_audit.json --imputer-artifacts artifacts/imputers --router-artifact artifacts/router_folds --forecaster-id chronos2

python -m tsfm_fais evaluate --config configs/smoke.yaml --impute-artifact artifacts/main-impute-chronos2 --forecaster-id chronos2 --forecaster-artifact checkpoints/chronos2 --output-dir artifacts/main-evaluate-chronos2
python -m tsfm_fais evaluate --config configs/smoke.yaml --impute-artifact artifacts/main-impute-chronos2 --forecaster-id chronos2 --forecaster-artifact checkpoints/chronos2 --output-dir artifacts/main-evaluate-chronos2 --resume
python -m tsfm_fais summarize --input artifacts/main-evaluate-chronos2 --output-dir artifacts/main-summary-chronos2
python -m tsfm_fais summarize-main --input artifacts/main-eval-chronos2-v1 artifacts/main-eval-timesfm2p5-v1 artifacts/main-eval-chronosbolt-v1 artifacts/main-eval-sundial-v1 artifacts/main-eval-tirex-v1 --output-dir artifacts/main-summary-v1 --bootstrap-replicates 2000 --bootstrap-seed 20260710

python -m tsfm_fais smoke --config configs/smoke.yaml
```

`data audit` 会实际读取 manifest 中启用的数据，并可将逐数据集审计结果和内容 SHA-256 写入指定 JSON。仓库不复制数据；当前 manifest 指向相邻仓库，在其他机器上运行前需要调整路径。四个 `run` stage 默认检查各自所需的前置产物，写入审计清单并以 `STAGE PREPARED` 正常结束；只有追加 `--execute` 才会执行训练或推理。命令不会自行下载数据或模型。

`smoke` 要求显式提供 `--config`，安装后的命令不会假定当前目录包含源码仓库的 `configs/`。

只有在用户明确追加 `--execute` 时，阶段执行器才会开始工作：`fit-imputers` 在每条轨迹前部的拟合区间内按 horizon 步长构造滚动窗口并序列化候选；`labels` 从拟合区间之后生成滚动 episode、单块标签和稳定种子抽样的块对标签；`train-router` 训练两个 LambdaMART 排序器与 Huber 块对模型；`impute` 加载冻结 artifact 和路由器并保存逐 episode 填补结果与块分配。块对抽样优先覆盖不同边及候选的左右角色，尾部块会排除声明为不支持尾部填补的候选。可用预测起点按时间顺序划分，较早的 70% 只用于教师标签与路由训练，较晚的 30% 只用于 `impute` 评估；两阶段不会复用同一个预测起点。执行器要求本地审计、权重和上游产物，不包含自动下载逻辑。

`main-v1` 的实际阶段关系可用下面的命令骨架复现。`labels` 需要分别对五个模型设置 `$MODEL` 与 `$CHECKPOINT` 后执行；四个独立单变量 TSFM 共用 `timesfm2p5` 预测模式生成的填补产物，但最终 `evaluate` 仍分别加载自己的 checkpoint。

```powershell
python -m tsfm_fais run --config configs/main.yaml --stage fit-imputers --run-id main-fit-v1 --audit-artifact artifacts/data_audit.json --execute
$MODEL = 'timesfm2p5'
$CHECKPOINT = 'checkpoints/timesfm2p5'
python -m tsfm_fais run --config configs/main.yaml --stage labels --run-id "main-labels-$MODEL-v1" --audit-artifact artifacts/data_audit.json --imputer-artifacts artifacts/main-fit-v1/imputer_artifacts --forecaster-id $MODEL --forecaster-artifact $CHECKPOINT --execute
python -m tsfm_fais labels merge --inputs artifacts/main-labels-chronos2-v2 artifacts/main-labels-timesfm2p5-v1 artifacts/main-labels-chronosbolt-v1 artifacts/main-labels-sundial-v1 artifacts/main-labels-tirex-v1 --output-dir artifacts/main-labels-merged-v1
python -m tsfm_fais run --config configs/main.yaml --stage train-router --run-id main-router-v1 --labels-artifact artifacts/main-labels-merged-v1/teacher_labels.jsonl --execute
python -m tsfm_fais run --config configs/main_eval.yaml --stage impute --run-id main-impute-joint-v1 --audit-artifact artifacts/data_audit.json --imputer-artifacts artifacts/main-fit-v1/imputer_artifacts --router-artifact artifacts/main-router-v1/router_folds --forecaster-id chronos2 --execute
python -m tsfm_fais run --config configs/main_eval.yaml --stage impute --run-id main-impute-univariate-v1 --audit-artifact artifacts/data_audit.json --imputer-artifacts artifacts/main-fit-v1/imputer_artifacts --router-artifact artifacts/main-router-v1/router_folds --forecaster-id timesfm2p5 --execute
python -m tsfm_fais evaluate --config configs/main_eval.yaml --impute-artifact artifacts/main-impute-joint-v1 --forecaster-id chronos2 --forecaster-artifact checkpoints/chronos2 --output-dir artifacts/main-eval-chronos2-v1
python -m tsfm_fais evaluate --config configs/main_eval.yaml --impute-artifact artifacts/main-impute-univariate-v1 --forecaster-id $MODEL --forecaster-artifact $CHECKPOINT --output-dir "artifacts/main-eval-$MODEL-v1"
```

`labels` 的 `--forecaster-id` 接受单个 ID，也接受以逗号分隔的多个 ID。单模型调用可以直接把本地 checkpoint 文件或目录传给 `--forecaster-artifact`。多模型调用应传入一个目录，其中每个模型位于以模型 ID 命名的子路径；也可以传入 JSON 映射文件，键为模型 ID，值为对应 checkpoint 路径，相对路径按映射文件所在目录解析。例如：

```json
{
  "chronos2": "chronos2",
  "timesfm2p5": "timesfm2p5"
}
```

`leave_dataset_out` 和 `leave_model_out` 会在 `train-router` 产物中写入 `router_folds/folds.json`。`impute --router-artifact <router_folds>` 会读取该索引：前者按当前数据的 `family_id` 选择相应留出折，后者按 `--forecaster-id` 选择相应留出折。也可以直接传入某个折的目录；执行器会核验其 `held_out` 元数据，防止把该折用于其他数据族或预测器。`rolling_origin` 仍使用单个 `router` 目录。

`impute` 的每个 NPZ 会保存 B-FAIS 填补上下文、原始观测掩码、clean context、clean future、候选张量、原生有效掩码和资源测量。`save_all_candidate_outputs: true` 用于最终评估：它在路由结果确定后补算其余可用候选，但不会改变 B-FAIS 的短名单或块分配。B-FAIS 的运行时间覆盖候选预筛、伪缺失、路由和组装；单候选时间只覆盖该填补器的一次推理。内存字段 `rss_delta_bytes` 是进程端点 RSS 正增量估计，不表示真实峰值。

`evaluate` 加载指定的本地 TSFM checkpoint，对 clean context、B-FAIS、作为 missing anchor 的 `locf`、`linear_interp` 和所有原生有效且能力匹配的候选执行预测；oracle 定义为该 episode 中下游 MASE 最低的有效单算法候选。不支持尾部填补或原生输出无效的候选会保留诊断行，但不会调用 TSFM，也不会进入指标均值。汇总结果同时报告 `count`、`metric_count`、`invalid_count` 和 `invalid_rate`。MASE 使用 clean context：上下文长度大于数据周期时采用季节 lag，否则回退到 lag 1。

逐 episode、逐方法结果写入 `episode_metrics.jsonl` 和 `episode_metrics.csv`。指标包括目标列宏平均 MASE、MAE、RMSE、缺失位置的填补 MAE/RMSE、`degradation_vs_clean_mase`、相对 clean 退化，以及相对 oracle regret。四个 `independent_univariate` 预测器可以复用同一份独立单变量模式的 impute 产物；`chronos2` 必须使用 `joint_multivariate` 产物。每行同时记录实际评测预测器 `forecaster_id` 和路由来源 `routing_forecaster_id`。

`--resume` 根据 `forecaster_id + episode_id + method` 跳过已完成预测，并可截断异常退出留下的不完整 JSONL 尾行。恢复前还会校验输入填补产物、checkpoint 路径、上下文、horizon、目标列、采样数、随机种子和设备；签名不一致时必须使用新的输出目录。

`summarize` 流式读取单个评估 JSONL，默认按数据集、数据族、预测器、缺失机制、缺失率和方法分组，基于有效行输出均值、样本标准差、最小值和最大值到 `summary.json` 与 `summary.csv`，同时保留完整的有效性计数。`summarize-main` 合并多个已完成的预测器评估，生成同 episode 配对比较、分组比较、百分位 bootstrap 区间和 Markdown 报告。评估和汇总不会自动下载模型；`--forecaster-artifact` 必须指向本地文件或目录。

产物目录约定为 `artifacts/<run_id>/`。当前阶段入口会写入 `resolved_config.json`、`software_versions.json`、`seeds.json`、`candidate_status.json` 和 `stage_manifest.json`；依赖缺失与“已准备、未启动执行”状态都会被持久化。真实阶段完成后还应在同一目录写入候选 artifact、教师标签、路由模型、块分配与回退记录。公共 `tsfm_fais.routing.RouterBundle.save()` 会保存 `router_bundle.joblib` 以及特征 schema、候选清单和依赖版本清单；`RoutingResult` 明确保存候选成本、启用成本、风险能量、成本能量和逐块回退记录；候选 runner 返回耗时、RSS 增量估计、状态和失败原因。`artifacts/`、数据、模型权重、缓存和结果目录已排除在版本控制之外。

## 防止未来信息泄漏

每个 rolling-origin episode 必须严格按下面的顺序构造：

1. 先确定预测起点 `t₀`、上下文长度和 horizon。
2. 将输入截断为 `[t₀-L,t₀)`；填补器和路由器只接收这段历史。
3. 只在截断后的完整历史中注入缺失，或读取其中已有的真实缺失掩码。
4. 伪缺失证据只在仍然观测的历史位置上构造。
5. 填补候选使用训练折拟合的冻结 artifact；评估 episode 不更新模型或标准化器。
6. 路由完成后才调用 TSFM；`[t₀,t₀+H)` 仅用于计算教师损失或最终评估。
7. 数据切分按 `family_id` 分组，避免同源版本跨训练与评估集合。
8. 同一轨迹的候选预测起点按时间顺序切分；较早起点生成教师标签，较晚起点用于最终填补评估。

`build_episode` 已把上述时间顺序固化为代码：先切出 clean context 和 future，再对 context 注入缺失。测试会修改预测起点之后的数据并确认缺失上下文和随机种子不受影响。

## 测试与本地验收

核心单元测试覆盖数据契约、CSV 多 item 边界、严格数据审计、六种缺失机制、稳定种子与预测起点隔离、同族/留一模型切分、原子块和关系图、20 个候选 ID、轻量候选的形状和观测值保持、结构化候选失败状态、代表性 PyPOTS 延迟导入、构造签名、mock 生命周期与采样输出、预测展开/重组、单块教师标签、固定 MASE 尺度和块对交互公式、短名单、预算、beam 与穷举一致性，以及失败候选的显式回退。

集成 smoke 使用内置 `48×3` 完整周期序列、一个内部块、一个尾部块、三个轻量候选，以及联合多变量和独立单变量 mock forecaster。它不访问网络，不下载权重，不使用 GPU，也不训练真实路由器。

```powershell
python -m compileall src tests
python -m pytest -q -m "not slow and not gpu and not network"
python -m tsfm_fais smoke --config configs/smoke.yaml
```

smoke 成功时打印 `SMOKE PASS`。`slow`、`gpu` 和 `network` marker 用于隔离后续需要真实依赖或外部资源的测试。

2026-07-14 的本地验收结果为：`compileall` 通过，非慢速/非 GPU/非网络测试 `227 passed`，离线 smoke 输出 `SMOKE PASS`。测试中的一条 MICE `ConvergenceWarning` 来自全缺失训练通道边界用例，不影响测试结论。

## 主实验结果（`main-v1`）

主实验使用 `configs/main.yaml` 生成教师标签与路由器，使用 `configs/main_eval.yaml` 进行最终评估。设置为上下文长度 48、预测长度 8、目标列 `[0,1]`、六种缺失机制、缺失率 `{0.1,0.2,0.3,0.5}` 和评估种子 `{22,33,44}`。每个数据版本抽取 24 个最终 episode，共 768 个底层缺失 episode。Chronos-2 使用联合多变量预测；TimesFM 2.5、Chronos-Bolt、Sundial 和 TiRex 使用独立单变量预测。每个预测器评估 16,896 个“episode—方法”组合，五个预测器合计 84,480 行。

候选池保留 20 个注册 ID，其中 TRMF 因当前 PyPOTS 1.5 接口与冻结 artifact 协议不兼容而未执行，主实验实际评估 19 个候选。候选拟合阶段记录 384 个数据集级已拟合 artifact、224 个无状态候选状态和 32 个 TRMF 不可用状态。五个教师任务合计生成 40,905 条单块标签和 7,620 条块对标签；路由训练完成 19 个 leave-family-out 折。联合与独立单变量两套填补评估均完成 768/768 episode，共覆盖 186,125 个缺失单元和 39,906 个极大缺失块。严格产物审计确认所有输出有限、原观测值逐元素保持、最终块值与分配候选一致，且没有选中无效候选或触发块回退。

下表报告同一预测器、同一 episode 的配对 MASE。差值定义为 `B-FAIS − 对照`，越小越好；区间为 2,000 次 episode 级百分位 bootstrap 的 95% 描述性区间。

| 对照 | 有效配对 | 配对 B-FAIS MASE | 对照 MASE | MASE 差值 [95% 区间] | B-FAIS 严格胜率 |
|---|---:|---:|---:|---:|---:|
| clean context | 3,840/3,840 | 12.7497 | 10.4970 | +2.2528 [1.1998, 3.2880] | 29.82% |
| LOCF | 3,685/3,840 | 12.3328 | 11.2259 | +1.1070 [0.0595, 2.0398] | 39.59% |
| linear interpolation | 1,945/3,840 | 10.1407 | 10.4491 | −0.3084 [−1.6761, 0.8461] | 39.02% |
| single-imputer oracle | 3,840/3,840 | 12.7497 | 8.7088 | +4.0409 [2.9427, 5.3629] | 5.00% |

当前结果不支持“B-FAIS 优于简单基线”的结论。B-FAIS 实现了 100% episode 覆盖，在 12 个同样全覆盖的方法中具有最低的总体平均 MASE，但其 episode 内平均名次为 6.11/12，且相对 LOCF 的配对差值为正。linear interpolation 只在 50.65% 的 episode 原生有效，表中的轻微均值优势区间跨 0，并且 19 个数据族等权后差值转为 `+0.1199`。四个独立单变量 TSFM 上，B-FAIS 相对 clean、LOCF 和 linear 的差值区间均位于 0 以上；Chronos-2 的对应区间跨 0。Chronos-2 同时是唯一的联合多变量预测器，因此本实验无法分离预测模式与模型身份的影响。

主要路由异常是对 CSDI 和 SoftImpute 的集中分配：两者合计占联合路由 66.48% 的块、独立单变量路由 64.20% 的块，而它们的总体 MASE 分别为 32.1948 和 29.1598。缺失率从 0.1 增至 0.5 时，B-FAIS MASE 从 9.9840 增至 16.8483；尾部混合与数值依赖缺失也是主要退化来源。MASE 分布高度偏斜，约 1% 的最大 episode 贡献 B-FAIS MASE 总和的 72.52%，其中 `Coastal_T_S_20T + Chronos-2` 受到极小 MASE 缩放分母影响。移除该数据版本后，B-FAIS 相对 clean、LOCF 和 linear 的平均差仍为 `+1.3267/+0.9632/+0.1758`，因此极端值会改变幅度，但不能改变总体诊断。

完整结果位于 `artifacts/main-summary-v1/`：`report.md` 给出方法、预测器、数据族、机制和缺失率分组结果，`method_summary.csv` 与 `comparison_summary.csv` 保存机器可读表，`main_summary.json` 保存输入 SHA-256 和汇总元数据。`artifacts/` 默认被忽略，因此这些路径只在完成本地主实验的工作区中存在。Chronos-Bolt 的一次全新评估重跑与原始 JSONL、CSV 字节级一致；联合模式的 `--resume` 复核复用了 768/768 个已提交 episode，且输入与输出哈希未变化。

## 当前限制

首轮主实验只训练了一次路由器，没有估计路由训练随机性。当前 95% 区间按 episode 重采样，没有按 item、预测起点或数据族聚类；3,840 次预测评估来自 768 个底层 episode 在五个 TSFM 上重复评估，不能视为 3,840 个完全独立样本。汇总包含 735 个比较分组且没有多重比较校正，因此区间只用于描述，不作确认性显著性结论。linear interpolation 和 seasonal lag 的原生有效率分别只有 50.65% 和 4.56%，相关比较受到可用性筛选影响。跨数据族的 MAE、RMSE 和填补误差还会混合不同量纲，应优先结合族内结果解释。

TRMF 当前不可执行。MICE 在完整训练窗口上拟合时缺少真实缺失模式，当前 artifact 实际提供接近单轮链式回归的行为。TimeMixer++ 在两次独立 GPU 填补运行中有 14/768 个 episode 出现小幅数值差异，最大绝对差约 `9.3e-4`；其他已做的恢复与重跑检查通过。标签合并后每个教师模型各有一个单候选 ranking group，LightGBM 可以读取，但该 group 不产生候选对排序信息。第三方候选调用期间尚未实施抢占或硬中止。

当前评估使用完整数据上的人工缺失注入，尚未验证自然缺失场景。结果明确显示当前路由器需要重新校准，后续改动应优先抑制 CSDI/SoftImpute 的过度分配，改进高缺失率与尾部块特征，并使用数据族级重采样和稳健尺度指标复核。任何改动后的性能结论都需要使用新的训练产物和独立评估目录，不能覆盖或选择性复用 `main-v1` 结果。

## 可核验来源

以下链接用于核验外部算法接口、预测模型和公共数据来源；本仓库中的具体默认参数仍以代码和 YAML 为准。

- 多变量缺失值工具与深度模型接口：[PyPOTS 1.5 文档](https://docs.pypots.com/en/stable/pypots.html)
- KNN、IterativeImputer 与缺失值处理：[scikit-learn Imputation 文档](https://scikit-learn.org/stable/modules/impute.html)
- 状态空间模型：[Statsmodels State Space 文档](https://www.statsmodels.org/stable/statespace.html)
- SVD 实现：[SciPy `linalg.svd` 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)
- LambdaMART 与回归器接口：[LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- Arrow IPC 格式：[Apache Arrow Python IPC 文档](https://arrow.apache.org/docs/python/ipc.html)
- Chronos-2 与 Chronos-Bolt：[Amazon Chronos 官方仓库](https://github.com/amazon-science/chronos-forecasting)
- TimesFM：[Google Research TimesFM 官方仓库](https://github.com/google-research/timesfm)
- Sundial：[Sundial 论文](https://arxiv.org/abs/2502.00816)
- TiRex：[NX-AI TiRex 官方仓库](https://github.com/NX-AI/tirex)
- 公共预测数据归档：[Monash Forecasting Repository](https://forecastingdata.org/)
- ETT 数据：[ETDataset 官方仓库](https://github.com/zhouhaoyi/ETDataset)

本地准入集合的唯一权威清单是 [`configs/data/datasets.yaml`](configs/data/datasets.yaml)。`main-v1` 的 32 个条目均已通过本地审计；修改文件、路径、哨兵或 schema 后必须重新生成审计结果。
