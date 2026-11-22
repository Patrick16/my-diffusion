# Starts the my-diffusion HTTP API and the web UI, each in its own console window.
#
# Why separate windows: the logs of both stay visible, and Ctrl+C or closing a window
# stops only that process.
#
#   pwsh scripts/start.ps1                       # API on 127.0.0.1:8000, web UI on 127.0.0.1:8080
#   pwsh scripts/start.ps1 -WebHost 0.0.0.0      # web UI for the whole local network (see clients/webui/README.md)
#   pwsh scripts/start.ps1 -NoWeb                # API only
#   pwsh scripts/start.ps1 -NoApi                # web UI only (the API must be running elsewhere)

param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [string]$WebHost = "127.0.0.1",
    [int]$WebPort = 8080,
    [switch]$NoApi,
    [switch]$NoWeb
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# uv is required for both projects
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not found in PATH. Install it with: winget install --id astral-sh.uv -e"
}

# Use PowerShell 7 for the new windows when it is installed, otherwise Windows PowerShell
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }

function Start-Service-Window {
    param(
        [string]$Title,
        [string]$Command,
        [string]$WorkingDirectory
    )
    # -NoExit keeps the window open after the process ends, so an error message can be read
    $inner = "`$Host.UI.RawUI.WindowTitle = '$Title'; $Command"
    Start-Process -FilePath $shell `
        -ArgumentList @("-NoExit", "-NoProfile", "-Command", $inner) `
        -WorkingDirectory $WorkingDirectory | Out-Null
    Write-Host "started: $Title" -ForegroundColor Green
}

if (-not $NoApi) {
    $models = Join-Path $root "models"
    if (-not (Test-Path $models)) {
        Write-Warning "models/ folder not found. Run scripts/link_models.ps1 first, otherwise generation fails."
    }
    Start-Service-Window -Title "my-diffusion API" `
        -Command "uv run uvicorn my_diffusion.api:app --host $ApiHost --port $ApiPort --timeout-graceful-shutdown 5" `
        -WorkingDirectory $root
    Write-Host "  API:    http://${ApiHost}:$ApiPort/docs"
}

if (-not $NoWeb) {
    Start-Service-Window -Title "my-diffusion web UI" `
        -Command "uv run --project clients/webui my-diffusion-webui --host $WebHost --port $WebPort" `
        -WorkingDirectory $root
    Write-Host "  Web UI: http://${WebHost}:$WebPort"
}

Write-Host "Close the windows (or press Ctrl+C in them) to stop the services."
