---
name: codex-iteration-runner
description: 搭建、审查或运行无人值守的 Claude/Codex 仓库迭代流程。当用户想把迭代脚本封装成 Codex skill、安装或分享周期性迭代工作流、配置统一 invoke_iteration.ps1 入口、把 research-writing-skill 门槛写入迭代提示词，或判断迭代脚本是否安全可分享时使用。
---

# Codex 迭代运行器

使用这个 skill 时，目标是把仓库里的“单轮代理脚本”整理成可以重复触发的迭代流程。这个 skill 本身不会无限运行；它负责帮助 Codex 安装、审查和操作由 Windows 任务计划程序或用户手动触发的 PowerShell 入口。

这个 skill 还内置通用审稿人式迭代提示词模板。安装后，Codex 不只知道怎么跑脚本，也知道一个论文迭代 prompt 应该包含哪些审稿标准、证据边界、图文要求和验证门槛。但具体论文路径、研究问题、实验结果和贡献定位仍必须由目标仓库自己的 `scripts/iteration/iterate_prompt.md` 提供。

## 核心流程

1. 找到仓库根目录，并检查是否存在迭代脚本目录。优先使用 `scripts/iteration/` 和统一入口 `scripts/iteration/invoke_iteration.ps1`。
2. 确认当前仓库使用的运行器类型：
   - Claude 运行器：`scripts/iteration/invoke_iteration.ps1 -Runner claude`
   - Codex 运行器：`scripts/iteration/invoke_iteration.ps1 -Runner codex`
   - 迭代提示词：`scripts/iteration/iterate_prompt.md`
   - 日志解析器：`scripts/iteration/parse_iter_stream.py` 和 `scripts/iteration/parse_codex_stream.py`
3. 读取内置提示词资产，并用它们生成或审查项目自己的 `iterate_prompt.md`：
   - 通用迭代模板：`templates/iterate_prompt.template.md`
   - 成熟项目提示词样例：`templates/iterate_prompt_example.md`
   - 审稿人门槛：`references/reviewer-gates.md`
   - `research-writing-skill` 使用门槛：`references/research-writing-gates.md`
   - 可迁移性清单：`references/portable-iteration.md`
4. 在建议分享之前，先检查可迁移性：
   - 提示词和设置文档中不能有未标注为示例的硬编码个人路径。
   - 不能包含 secret、完整 API key、personal access token 或私有 base URL。
   - 项目专用提示词必须明确标注为“别人使用前需要重写”。
   - 运行脚本应当根据脚本自身位置定位 `REPO_ROOT`。
   - 缺少必要 CLI、远程仓库、日志解析器或写作 skill 时，运行脚本应当快速失败。
5. 如果仓库是论文写作项目，确认提示词和运行脚本都要求使用 `Norman-bury/research-writing-skill`。路径可以来自 `RESEARCH_WRITING_SKILL_REPO`，也可以来自常见本地安装目录。如果读取不到该 skill，脚本应当在修改论文前停止。
6. 真正运行迭代前，先执行不会改动论文的验证命令：

```powershell
python -m py_compile `
  .\scripts\iteration\run_iteration.py `
  .\scripts\iteration\parse_iter_stream.py `
  .\scripts\iteration\parse_codex_stream.py
```

7. 只有用户明确要求时，才运行真实迭代。真实迭代可能会拉取代码、修改文件、提交、推送，并消耗 API 额度。

## 提示词模板使用方式

当用户要求“把这个迭代逻辑给别人复用”或“为新仓库安装迭代 skill”时，按下面的规则处理：

- 如果目标仓库还没有 `scripts/iteration/iterate_prompt.md`，先基于 `templates/iterate_prompt.template.md` 生成一份项目专用提示词。生成时必须替换所有占位符，例如论文路径、研究主线、证据文件、禁止性表述、编译命令和 sanity 命令。
- 生成新项目提示词时，可以按需读取 `templates/iterate_prompt_example.md`。这个文件是成熟项目提示词样例，适合借鉴章节组织、审稿强度、证据边界、验证门槛和 commit audit 写法。
- `iterate_prompt_example.md` 只能作为写法参考，不能复制其中的项目事实、论文数量、路径、研究定位、实验数字、禁用词列表或任务计划名称。新项目必须把这些内容替换为自己的真实材料。
- 如果目标仓库已经有 `iterate_prompt.md`，不要直接覆盖。先用 `references/reviewer-gates.md` 和 `references/research-writing-gates.md` 审查它是否包含审稿人视角、证据边界、claim gate、图文统一、去 AI 味、验证命令和 commit 要求。
- 不要把某个项目的具体论文定位写进 skill 默认模板。模板只能提供审稿流程和质量门槛；项目事实必须留在项目自己的 prompt 里。
- 如果用户要迁移其他仓库经验，可以把旧提示词中的高层质量标准作为示例段落，但必须明确“这是示例，不能原样用于新项目”。

## 搭建原则

创建或修复迭代流程时，保持下面这些边界：

- `invoke_iteration.ps1` 只做一轮，然后退出。
- 周期性触发放在脚本外部，例如 Windows 任务计划程序或用户自己管理的外部循环。
- 如果已有自动化依赖旧路径，优先修改外部任务配置指向 `scripts/iteration/invoke_iteration.ps1`，避免继续维护薄包装脚本。
- 真实脚本放在 `scripts/iteration/` 下，便于分享和维护。
- 提示词必须清楚写出证据边界、claim gate、编译命令和必需 skill。
- 分享说明中要明确：迭代逻辑可以复用，但认证信息、提示词、仓库路径、远程仓库、模型名称和定时配置都必须按新项目调整。

## 安全检查

在宣称脚本“可以分享”或“已经安装好”之前，运行：

```powershell
python -m py_compile `
  .\scripts\iteration\run_iteration.py `
  .\scripts\iteration\parse_iter_stream.py `
  .\scripts\iteration\parse_codex_stream.py

rg -n "sk-|github_pat_|ANTHROPIC_API_KEY=.*[A-Za-z0-9]|OPENAI_API_KEY=.*[A-Za-z0-9]" .\scripts .\scripts\iteration
rg -n "C:\\Users\\|D:\\|E:\\|<owner>|<repo>" .\scripts .\README.md
git status --short scripts scripts/iteration
```

解释结果时要区分情况：占位符和示例路径可以接受；可执行脚本里的个人绝对路径通常不应该保留。

## 当前仓库结构

这个仓库推荐的布局是：

```text
scripts/
  iteration/
    run_iteration.py           统一运行器
    invoke_iteration.ps1       Claude/Codex 共享 PowerShell 启动器
    iterate_prompt.md          当前项目专用提示词
    parse_iter_stream.py
    parse_codex_stream.py
    README.md
    codex-iteration-runner/
      SKILL.md
      templates/
        iterate_prompt.template.md
        iterate_prompt_example.md
      references/
        portable-iteration.md
        reviewer-gates.md
        research-writing-gates.md
```

如果用户要求“安装这个 iteration skill”，把该 skill 目录复制到：

```powershell
$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
$SkillRoot = Join-Path $CodexHome "skills"
New-Item -ItemType Directory -Force -Path $SkillRoot
Copy-Item -Recurse -Force `
  -LiteralPath ".\scripts\iteration\codex-iteration-runner" `
  -Destination (Join-Path $SkillRoot "codex-iteration-runner")
```

然后提醒用户重启 Codex，让新 skill 在后续会话中生效。

## 参考资料

- 可迁移性检查清单见 `references/portable-iteration.md`。
- 审稿人式迭代门槛见 `references/reviewer-gates.md`。
- `research-writing-skill` 使用门槛见 `references/research-writing-gates.md`。
- 通用项目提示词模板见 `templates/iterate_prompt.template.md`。
- 成熟项目提示词样例见 `templates/iterate_prompt_example.md`；只用于借鉴结构和审查标准。
