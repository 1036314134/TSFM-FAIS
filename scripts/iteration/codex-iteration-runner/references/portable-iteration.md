# 可迁移迭代清单

把这套迭代流程迁移到另一个仓库时，使用本清单检查。当前推荐只保留一个 PowerShell 入口，通过参数选择 Claude 或 Codex，避免维护多层包装脚本。

## 推荐布局

```text
scripts/
  iteration/
    invoke_iteration.ps1
    run_iteration.py
    iterate_prompt.md
    parse_iter_stream.py
    parse_codex_stream.py
    README.md
```

`scripts/iteration/invoke_iteration.ps1` 负责选择 Python 并把 `-Runner claude|codex` 交给 `run_iteration.py`。如果旧任务计划仍调用 `scripts/iterate_once*.ps1`，优先更新任务计划，不建议继续保留薄包装脚本。

## 必需输入

- 一个 git 仓库，并且 `origin` 远程仓库可写。
- 一个每次执行后都会退出的单轮运行器。
- 一个项目专用 `scripts/iteration/iterate_prompt.md`。
- 位于运行器外部的触发机制，例如 Windows 任务计划程序。
- 对应代理 CLI 的认证配置。Claude 运行器需要 `claude`，Codex 运行器需要 `codex`。

`iterate_prompt.md` 必须按目标仓库重写，至少写清楚目标论文路径、权威源文件、禁止读取为事实的文件、允许和禁止的 claim、必需 skill、验证命令、提交预期和剩余风险汇报要求。论文写作项目必须要求在修改前读取相关本地 `SKILL.md`；如果找不到对应 skill，脚本应停止。

## 运行与调度

手动运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\iteration\invoke_iteration.ps1 -Runner codex
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\iteration\invoke_iteration.ps1 -Runner claude
```

Windows 任务计划程序示例：

```powershell
$Repo = "<absolute path to target repo>"
$TaskName = "<project name> Codex Iterate"
$Action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Repo\scripts\iteration\invoke_iteration.ps1`" -Runner codex" `
  -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger `
  -Once `
  -At (Get-Date).AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Minutes 30)
$Settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew
Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $Action `
  -Trigger $Trigger `
  -Settings $Settings `
  -Description "Run one Codex paper iteration."
```

不要把定时触发放进 Codex skill。skill 负责创建或审查运行器和配置文件，周期性执行交给操作系统或外部任务工具。

## 分享前审查

分享前运行：

```powershell
python -m py_compile `
  .\scripts\iteration\run_iteration.py `
  .\scripts\iteration\parse_iter_stream.py `
  .\scripts\iteration\parse_codex_stream.py

rg -n "sk-|github_pat_|ANTHROPIC_API_KEY=.*[A-Za-z0-9]|OPENAI_API_KEY=.*[A-Za-z0-9]" .\scripts .\scripts\iteration
rg -n "C:\\Users\\|D:\\|E:\\|<owner>|<repo>" .\scripts .\README.md
git status --short scripts scripts/iteration
```

占位符和明确标注的示例路径可以保留；可执行脚本、项目提示词和本地配置文档中不应保留作者个人路径或真实 secret。模型名称、API 端点、账号权限、任务计划名称和运行预算都必须由使用者按自己的环境配置。
