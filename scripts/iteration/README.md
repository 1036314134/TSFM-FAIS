# 迭代脚本说明

这个目录只保留 TSFM-UA-MI 的单轮论文迭代入口和必要说明。脚本每次只执行一轮；定时触发、循环运行和用户环境配置由 Windows 任务计划程序或外部任务系统负责。

## 统一入口

本仓库不再保留 `scripts/iterate_once.ps1`、`scripts/iterate_once_codex.ps1`、`scripts/iteration/iterate_once.ps1` 和 `scripts/iteration/iterate_once_codex.ps1`。手动运行或任务计划程序都应直接调用统一入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\iteration\invoke_iteration.ps1 -Runner codex
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\iteration\invoke_iteration.ps1 -Runner claude
```

`invoke_iteration.ps1` 负责选择 Python 并调用 `run_iteration.py --runner claude|codex`。实际的进程管理、日志、锁、超时、`git pull --rebase` 和脏工作区处理都在 `run_iteration.py` 中。

## 文件边界

- `invoke_iteration.ps1`：Claude/Codex 共用的 PowerShell 启动器。
- `run_iteration.py`：统一单轮运行器。
- `iterate_prompt.md`：当前项目专用提示词，不能作为通用模板迁移到其他仓库。
- `parse_iter_stream.py`：Claude 流式日志解析器。
- `parse_codex_stream.py`：Codex JSONL 日志解析器。
- `codex-iteration-runner/`：可安装的 Codex skill 包。通用迁移说明集中在 `references/portable-iteration.md`。

## Windows 任务计划程序

下面示例创建 Codex 任务。Claude 版本只需要把任务名和 `-Runner codex` 改成 `-Runner claude`。

```powershell
$Repo = "<TSFM-UA-MI 仓库绝对路径>"
$TaskName = "TSFM-UA-MI Codex Iterate"
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
  -Description "Run one TSFM-UA-MI Codex iteration."
```

不要让 Claude 和 Codex 两个任务同时写同一个工作区。分享到其他仓库前，先重写 `iterate_prompt.md`，并参考 `codex-iteration-runner/references/portable-iteration.md` 检查认证、路径、远程仓库、任务名称和项目事实。

## 安装 Codex Skill

```powershell
$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME ".codex" }
$SkillRoot = Join-Path $CodexHome "skills"
New-Item -ItemType Directory -Force -Path $SkillRoot
Copy-Item -Recurse -Force `
  -LiteralPath ".\scripts\iteration\codex-iteration-runner" `
  -Destination (Join-Path $SkillRoot "codex-iteration-runner")
```

安装后需要重启 Codex，让 skill 注册表重新加载。安装 skill 只提供搭建和审查能力，不会替目标仓库生成项目事实或自动创建后台任务。
