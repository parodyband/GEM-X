<#
.SYNOPSIS
    Install GEM-X Live for Maya (Windows).

.DESCRIPTION
    One line, from PowerShell:

        irm https://github.com/parodyband/GEM-X/releases/latest/download/install.ps1 | iex

    or, with options:

        & ([scriptblock]::Create((irm https://github.com/parodyband/GEM-X/releases/latest/download/install.ps1))) -SkipModels

    Run from an extracted release zip (install.bat), it installs in place.

    What it does:
      1. Downloads the prebuilt bundle (Maya module, capture server, gem-x.cpp runtime).
      2. Installs OpenCV for Maya's own Python (mayapy) next to the server.
      3. Downloads the GEM-X models (about 4 GB, SHA-256 checked) unless -SkipModels.
      4. Registers the Maya module, so the GEM-X menu appears when Maya starts.

    Needs Maya 2025 or later and a GPU with a Vulkan driver.
#>
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "GEMX-Live"),
    [string]$Bundle = "",      # zip path or URL; default: the latest release
    [string]$Mayapy = "",      # default: the newest Maya 2025+ found
    [string]$ModulesDir = "",  # default: <Documents>\maya\modules
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is far faster without the progress bar
$Repo = "parodyband/GEM-X"
$Asset = "GEMX-Live-Maya-win64.zip"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`nGEM-X Live install failed: $text" -ForegroundColor Red; throw $text }

Write-Host "GEM-X Live for Maya installer" -ForegroundColor Green

# ---------------------------------------------------------------- Maya
Step "Finding Maya"
if (-not $Mayapy) {
    $candidates = @()
    if ($env:MAYA_LOCATION) { $candidates += Join-Path $env:MAYA_LOCATION "bin\mayapy.exe" }
    $candidates += Get-ChildItem "$env:ProgramFiles\Autodesk\Maya20*\bin\mayapy.exe" -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.FullName -replace '.*Maya(\d{4}).*', '$1') } -Descending |
        ForEach-Object { $_.FullName }
    foreach ($c in $candidates) {
        if ((Test-Path $c) -and ([int]($c -replace '.*Maya(\d{4}).*', '$1') -ge 2025)) { $Mayapy = $c; break }
    }
}
if (-not $Mayapy -or -not (Test-Path $Mayapy)) { Fail "Maya 2025 or later was not found. Pass -Mayapy <path to mayapy.exe>." }
Write-Host "  $Mayapy"

# ---------------------------------------------------------------- bundle
$local = $PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "runtime\gemx.dll")) -and -not $Bundle
if ($local) {
    $InstallDir = $PSScriptRoot
    Step "Installing in place: $InstallDir"
} else {
    Step "Downloading GEM-X Live"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("gemx-live-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp $Asset
        if (-not $Bundle) { $Bundle = "https://github.com/$Repo/releases/latest/download/$Asset" }
        if ($Bundle -match '^https?://') {
            Write-Host "  $Bundle"
            Invoke-WebRequest -Uri $Bundle -OutFile $zip -UseBasicParsing
        } else {
            Copy-Item $Bundle $zip
        }
        Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "x") -Force
        $src = Get-ChildItem (Join-Path $tmp "x") -Directory | Select-Object -First 1
        # Merge over an existing install; downloaded models and site-packages are kept.
        robocopy $src.FullName $InstallDir /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) {
            Fail "could not update $InstallDir. Close Maya (it may be using the capture server) and run the installer again."
        }
        $global:LASTEXITCODE = 0
    } finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  installed to $InstallDir"
}

# ---------------------------------------------------------------- Python dependency
Step "Installing OpenCV for Maya's Python"
$site = Join-Path $InstallDir "server\site-packages"
& $Mayapy -m pip install --disable-pip-version-check --no-warn-script-location --quiet `
    --no-deps --upgrade --target $site "opencv-python-headless>=4.9"
if ($LASTEXITCODE -ne 0) { Fail "pip could not install opencv-python-headless" }
& $Mayapy -c "import sys; sys.path.insert(0, r'$site'); import cv2, numpy; print('  OpenCV', cv2.__version__, '/ NumPy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "OpenCV does not import under mayapy" }

# ---------------------------------------------------------------- models
if ($SkipModels) {
    Step "Skipping models (download them later from the GEM-X Live window)"
} else {
    Step "Downloading models (about 4 GB, one time)"
    Write-Host "  GEM-X: NVIDIA Open Model License. ViTPose: DINOv3 License. YOLOX: Apache-2.0."
    Write-Host "  See $InstallDir\runtime\licenses before you ship anything made with them."
    & $Mayapy (Join-Path $InstallDir "server\fetch_models.py") --dest (Join-Path $InstallDir "runtime\models")
    if ($LASTEXITCODE -ne 0) { Fail "model download failed. Run the installer again to resume." }
}

# ---------------------------------------------------------------- Maya module
Step "Registering the Maya module"
$installArgs = @((Join-Path $InstallDir "install.py"))
if ($ModulesDir) { $installArgs += @("--modules-dir", $ModulesDir) }
& $Mayapy @installArgs
if ($LASTEXITCODE -ne 0) { Fail "could not write the Maya module file" }

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Start Maya, then GEM-X > Live Capture > Launch Server > Start."
