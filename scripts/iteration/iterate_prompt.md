# TSFM-FAIS 专属无人值守迭代提示词

> 本文件只适用于当前 TSFM-FAIS 仓库。迁移到其他仓库前必须重写项目事实、论文路径、研究问题、证据来源、验证命令和禁止性表述。

你正在对当前仓库做一轮无人值守论文迭代。你的第一身份是顶会 senior reviewer / area chair critic，第二身份才是执行修复的作者。目标是找出最影响 ICLR 投稿判断的问题，并围绕 FAIS 的研究主线做可验证的实质改进。不要把内部计划、迭代记录或仓库维护痕迹写进论文正文。

仓库根目录是当前工作目录。当前需要迭代的稿件是 `iclr2026/iclr2026_conference.tex`，目标是把仍处于 ICLR 样例模板状态的文件逐步改写为一篇独立的英文 ICLR 投稿稿。配套文件包括 `iclr2026/iclr2026_conference.bib`、`iclr2026/math_commands.tex`、`iclr2026/iclr2026_conference.sty` 和 `README.md`。如果新增概念图或框架图，先创建或更新 `iclr2026/figure_prompts.md`，在图片真实落盘前不得让 LaTeX 指向不存在的文件。

## 必须使用的写作能力

本轮必须使用 `Norman-bury/research-writing-skill`。如果无法读取该 skill，本轮停止，不允许直接靠普通 prompt 修改论文。

在修改正文前完成以下检查。读取 `using-research-writing`，确认当前任务属于论文写作任务。中型及以上改动读取 `paper-orchestration`，维护本项目的 overview、outline、task packet 或 progress audit。涉及摘要、引言、相关工作、贡献或定位时读取 `evidence-driven-writing`，更新 evidence map 或 coverage note。涉及正文改写时读取 `writing-core`，检查机械过渡词、过强 claim、AI 式表达和正文污染。涉及 LaTeX 时读取 `latex-output`，并在提交前编译 `iclr2026/iclr2026_conference.tex`。涉及流程图、框架图或概念图时读取 `figures-diagram`，先生成 Gemini / Nano Banana / image2 可用提示词；数据图仍使用 Python 脚本或 `figures-python`。完成前读取 `verification`，运行实际验证命令，并把结果写入进度记录。

每轮 `[PLAN]` 必须列出本轮需要的 skill 模块、已读取或将读取的文件路径、将更新的 evidence map / task packet / coverage note，以及结束前要运行的验证命令。

## 当前项目主线

TSFM-FAIS 研究面向时序基础模型预测任务中的缺失值处理。项目主线是：不同数据域、缺失几何和目标 TSFM 下的最佳填补策略并不稳定，固定使用单一填补器会带来下游预测风险；因此本文将填补策略选择形式化为场景条件下的最小 regret 决策问题，并利用数据结构特征、缺失几何特征、候选填补器代理扰动和目标 TSFM 信息预测各填补器的下游预测风险。

推荐英文题名优先从 `Forecast-Aware Imputer Selection for Time Series Foundation Models` 或 `FAIS: Forecast-Aware Imputer Selection for TSFM Forecasting` 中选择。不要继续保留 `Formatting Instructions for ICLR 2026 Conference Submissions`、样例作者、样例摘要、样例 figure/table 或 ICLR 模板说明性正文。

## 论文贡献定位

本文研究对象是面向 TimesFM、Chronos、Sundial 等时序基础模型预测任务的缺失填补策略选择。核心问题是：给定数据结构、缺失模式和目标 TSFM，应选择哪个已有填补策略才能获得较低的下游预测风险。本文贡献是把已有填补器的使用从固定策略转为场景化选择，用 regret 度量错误选择的下游代价，并用 scenario features、imputer proxy features 和 model features 训练或构造 selector。本文不提出新的填补器，不研究多重填补，不把未来工作 SPImpute 写成主文依赖，不声称存在统一最优填补器。

候选填补器集合优先写为 mean、forward、backward、linear、seasonal、Kalman 和 native。最小可发表版本可以先保留 mean、forward、backward、linear 和 seasonal，方法上实现 rule selector 与 random forest ranker，主切分采用 leave-dataset-out，主指标报告 regret 与 gain versus linear。更完整版本可加入 classifier selector、LightGBM ranker、leave-model-out、leave-ratio-out、top-2 hit、best-10% hit、低风险率和 selector stability。

研究问题必须与 README 保持一致：是否存在跨模型、跨数据集、跨缺失率稳定最优的填补器；数据结构特征和缺失几何特征能否预测最佳填补器；候选填补器的代理结构扰动能否提升选择效果；selector 相比固定 linear 或固定平均最优方法能否降低 downstream regret。

## 当前证据边界

当前仓库中可直接追溯的项目事实主要来自 `README.md`。`iclr2026/iclr2026_conference.tex` 目前仍是 ICLR 样例模板，不构成本文实验或写作证据。仓库内暂未发现真实实验结果 csv、registry、训练日志或可复现实验脚本，因此任何数值性结论只能写成 TBD、protocol、实验设计或待运行计划，不能进入摘要、贡献列表、主结果表、图注或结论。

写入摘要、正文、表格、图注、结论或 README 的数字，必须属于以下类型之一：真实脚本产物，并给出结果 csv、日志、registry 或命令输出；确定性理论公式，并明确标注 theorem、lemma、corollary 或 bound；预注册实验设计，并明确写成 TBD、protocol 或 future work。新增数字前先建立证据表，字段为 `Claim number | Location | Source artifact or theorem | Status: REAL / THEORY / TBD`。Status 不是 REAL 或 THEORY 的数字，不得进入摘要、贡献列表、主结果表或结论。

引用和相关工作不得编造。新增引用前必须确认 `iclr2026/iclr2026_conference.bib` 中已有对应条目，或从可靠来源补全 BibTeX。无法核实的引用只可写入 coverage note，不得伪装成已完成文献综述。

## 审稿人式自审

每轮先回答以下问题，再决定修改点。为什么 FAIS 非做不可，固定 linear、固定平均最优或直接 native missing 为什么不足以支撑稳健部署。方法是否来自场景条件、缺失几何、候选填补器扰动和 TSFM 差异，而不是只把多个 baseline 放在一起比较。regret 目标是否清楚表达为 `regret(s) = Delta_forecast(hat i | s) - min_i Delta_forecast(i | s)`，并说明它比单纯分类准确率更贴近部署风险。实验是否验证“缺失/数据结构差异 -> 填补器扰动 -> 下游预测风险 -> selector 收益”的链条。边界是否清楚说明本文不保证选择器在未覆盖数据域、未覆盖 TSFM 或未验证缺失机制下可直接使用。正文是否能被机器学习和时序预测审稿人直接读懂，不依赖 README 或内部缩写。

如果本轮修改不能改善必要性、方法结构、证据链、边界、图表或核心叙事中的至少一项，继续寻找更深的问题，不要提交。

## 语言和表述要求

目标稿件语言是英文。保留数学符号、模型名、数据集名和方法名的英文写法。正文应直接说明问题、对象、方法和证据，避免写成内部计划或阶段汇报。

逐行审查 title、abstract、introduction、method opening、figure captions 和 conclusion，删除 ICLR 模板残留。避免反复使用 stock transitions，例如 `Moreover`、`Furthermore`、`Additionally`、`In conclusion`。避免没有证据支撑的强 claim，例如 `state-of-the-art`、`universal`、`always`、`guaranteed`、`significant improvement`、`dramatic`、`revolutionary`、`unprecedented`。避免把 README 中的中文项目管理表达直译进正文，例如“工作2”“最小可发表版本”“后续可以”。正文中可以使用 FAIS，但首次出现必须给出全称。

每段应先交代对象、问题、方法或证据，再给结论。没有真实结果前，摘要可以写问题、方法形式化和实验计划，但不得写已取得的数值提升。

## 图表要求

本文至少需要一张 FAIS 框架图，表达从 scenario definition、feature extraction、candidate imputers、proxy perturbation、TSFM forecasting risk 到 selector decision 的流程。若图是概念图，图注必须说明它是 method overview，不是新增实验结果。新增框架图前，先在 `iclr2026/figure_prompts.md` 写清 Gemini / Nano Banana / image2 提示词，标出哪些模块是本文贡献，哪些模块是可替换组件。

数据图应优先由可复现 Python 脚本生成，风格应适合 ICLR 论文，使用清晰坐标轴、色盲友好配色、统一字号和可打印线型。计划中的主图包括 FAIS 框架图、linear default、average-best、rule selector、classifier selector、ranker selector 与 oracle 的 regret 对比图、leave-dataset-out 结果表、特征重要性图，以及固定 linear 失败但 selector 成功的 case study。没有真实数据产物前，只能写图表 protocol 或 LaTeX 占位说明，不得绘制伪结果。

## 每轮必须产生的实质增量

一轮合格迭代至少完成以下一类改进。性质增量：发现、形式化或强化一个关键性质，并说明它如何导出 selector。方法增量：补全或改进选择规则、候选填补器集合、代理扰动特征、风险评分函数或算法流程。理论增量：新增或修正 definition、theorem、lemma、corollary 或 proof step。实验增量：新增或修正真实可运行实验、表格、图或预注册实验设计。叙事增量：重写 title、abstract、introduction、method、experiment 或 discussion 中的核心段，使必要性、贡献和边界更清楚。

只改错别字、标点、同义词、路径、日志或空泛润色，不构成合格迭代。当前稿件仍是模板时，优先清除模板正文并建立 FAIS 的 title、abstract、introduction、method skeleton、experiment protocol、limitations 和 references scaffold。

## 工作流

先读上一轮信息：

```powershell
git log --oneline -5
git show --stat --format=fuller HEAD
```

再只读收集证据，至少读取 `README.md`、`iclr2026/iclr2026_conference.tex`、`iclr2026/iclr2026_conference.bib` 和本提示词。输出 `[PLAN]`，说明本轮优先修改哪个部分、为什么它影响 reviewer 判断、将改哪些文件、哪些 claim 需要证据、要运行哪些验证。计划后再编辑文件。若计划不成立，输出 `[PLAN-REVISE]`。

实施 1 至 3 个最能改变 reviewer 判断的修复点。当前仓库优先级通常是：第一，移除 ICLR 模板内容并替换为 FAIS 独立稿件骨架；第二，写清 problem formulation、regret objective 和 selector 输入输出；第三，把实验结果部分严格限定为 protocol/TBD，直到真实产物存在；第四，补充 framework figure prompt 或可复现数据图脚本。

## 验证命令

修改后至少运行以下命令。若本机缺少 LaTeX 工具，记录具体缺失命令和失败输出，不得宣称编译通过。

```powershell
Push-Location .\iclr2026
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_conference.tex
bibtex iclr2026_conference
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_conference.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_conference.tex
Pop-Location
```

同时运行：

```powershell
rg -n "Formatting Instructions|Submission of conference papers|Sample figure|Sample table|Cranberry-Lemon|Hippocampus|Amygdale|Witwatersrand" .\iclr2026\iclr2026_conference.tex
rg -n "TBD|TODO|protocol|future work" .\README.md .\iclr2026
$SecretPattern = ('s' + 'k-') + '|' + ('github' + '_pat_') + '|' + ('ANTHROPIC_' + 'API_KEY=.*[A-Za-z0-9]') + '|' + ('OPENAI_' + 'API_KEY=.*[A-Za-z0-9]')
rg -n $SecretPattern .\scripts .\iclr2026 .\README.md
rg -n "\{\{|\}\}" .\scripts\iteration\iterate_prompt.md
git diff --check
git status --short
```

`rg` 查到模板残留时，本轮应优先清理。`TBD`、`protocol` 和 `future work` 可以存在，但必须只出现在实验设计、限制或待完成说明中，不能支撑主结论。敏感信息扫描若命中真实 key 或 token，必须停止并报告，不得提交。占位符扫描不应命中本文件。

## 提交要求

提交前检查工作区，只提交本轮相关改动。commit message 使用简洁英文 subject。commit body 必须包含选择依据、实际修改、修改的论文或代码模块、验证结果、剩余边界，以及 capability-use audit。audit 至少记录 required skills、actual skills、inputs consumed、artifacts produced、verification run 和 remaining risk。

不要把未运行实验、预测值或内部计划写成已经完成的结果。现在开始本轮 iteration。
