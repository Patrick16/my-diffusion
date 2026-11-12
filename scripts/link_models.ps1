# Creates a models/ folder in the project with links (junctions) to the real model folders.
#
# Why: the project code only knows about ./models/... and does not depend on where the models are
# stored on a particular machine. The files themselves are not copied or downloaded.
#
# Why a junction and not a symlink: on Windows a junction does not require administrator rights
# or developer mode, and it works across different drives (models on D:, project on E:).
#
# On another machine: fix the paths below (or pass them as parameters) and run the script,
# or simply put the files into models/ with the same structure.
#
#   pwsh scripts/link_models.ps1
#   pwsh scripts/link_models.ps1 -ModelsHub "X:\models"

param(
    [string]$ModelsHub = "D:\ai\ModelsHub"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root "models"

# link inside the project -> real folder
$links = [ordered]@{
    "checkpoints"                          = "$ModelsHub\DiffusionModels\krea2"      # the Krea 2 model itself (transformer)
    "loras"                                = "$ModelsHub\Loras\krea2"
    "vae"                                  = "$ModelsHub\VAE\qwen"                   # qwen_vae.safetensors + qwen_vae_config.json
    "text_encoders\Qwen3-VL-4B-Instruct"   = "$ModelsHub\Wan2GP\Qwen3-VL-4B-Instruct"   # Krea 2 text encoder + enhancer by default
    "text_encoders\Qwen3_5_4B_Abliterated" = "$ModelsHub\Wan2GP\Qwen3_5_4B_Abliterated" # alternative enhancer
    "text_encoders\qwen3_8b"               = "$ModelsHub\Wan2GP\qwen3_8b"               # alternative enhancer
}

foreach ($name in $links.Keys) {
    $link = Join-Path $models $name
    $target = $links[$name]

    if (-not (Test-Path $target)) {
        Write-Warning "source folder not found, skipping: $target"
        continue
    }
    if (Test-Path $link) {
        $item = Get-Item $link -Force
        if ($item.LinkType -eq "Junction" -and $item.Target -eq $target) {
            Write-Host "ok      $name"
            continue
        }
        Write-Warning "$link already exists and it is not our link - leaving it alone"
        continue
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $link) | Out-Null
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    Write-Host "created  $name -> $target"
}
