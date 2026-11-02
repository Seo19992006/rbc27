param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid,

    [Parameter(Mandatory = $true)]
    [string]$Run3Directory
)

$ErrorActionPreference = 'SilentlyContinue'
$statusFile = Join-Path $PSScriptRoot 'stop_after_run2.status.txt'

while ($true) {
    $training = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    if (-not $training) {
        "Training process ended before run 3 began: $(Get-Date -Format o)" | Set-Content -LiteralPath $statusFile -Encoding ASCII
        exit 0
    }

    if (Test-Path -LiteralPath $Run3Directory -PathType Container) {
        "Run 2 completed; training process stopped before run 3: $(Get-Date -Format o)" | Set-Content -LiteralPath $statusFile -Encoding ASCII
        Stop-Process -Id $TrainingPid -Force
        exit 0
    }

    Start-Sleep -Seconds 2
}
