# Unattended entrypoint for Windows Task Scheduler (§17, §21: "nightly batch
# as a scheduled deployment"). Wraps `python -m stockpredictor.orchestration.
# nightly_flow` (already idempotent/resumable and already sends a Telegram
# alert on failure via monitoring/alerts.py) with the two things a
# *scheduled* run additionally needs that an interactive run doesn't:
#
#   1. A persisted log file -- Task Scheduler doesn't show console output,
#      and stdout-only logging (common/logging.py) is otherwise lost the
#      moment the process exits.
#   2. A propagated process exit code -- so a failed run shows up as
#      "failed" in Task Scheduler's own history too, not just via Telegram
#      (belt-and-suspenders: the Telegram alert can silently fail, e.g. if
#      the machine has no network at that moment).
#
# Deliberately no log-rotation *library* -- one run/day is a handful of KB;
# a 30-day retention loop below is simpler than adding a dependency for it
# (§16: don't add infra before a metric forces it).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Get-ChildItem $logDir -Filter "nightly_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force

$logFile = Join-Path $logDir ("nightly_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "venv not found at $python -- run 'python -m venv .venv; .venv\Scripts\pip install -e .[dev]' first."
    exit 1
}

& $python -m stockpredictor.orchestration.nightly_flow 2>&1 | Tee-Object -FilePath $logFile
exit $LASTEXITCODE
