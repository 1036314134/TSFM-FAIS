param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("claude", "codex")]
    [string]$Runner,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs = @()
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$PythonRunner = Join-Path $ScriptDir "run_iteration.py"

function Invoke-IterationPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$CommandArgs = @()
    )

    & $Command @CommandArgs $PythonRunner --runner $Runner @RunnerArgs
    exit $LASTEXITCODE
}

if ($env:PYTHON) {
    Invoke-IterationPython -Command $env:PYTHON
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    Invoke-IterationPython -Command $Python.Source
}

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PyLauncher) {
    Invoke-IterationPython -Command $PyLauncher.Source -CommandArgs @("-3")
}

throw "Python 3 is required. Install Python or set the PYTHON environment variable."
