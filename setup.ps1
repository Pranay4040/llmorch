<#
.SYNOPSIS
    Set llmorch up once, so it can be started like any other program.

.DESCRIPTION
    Nobody should have to remember a directory path to run a program. This does
    the four things that stand between a fresh clone and a double-click:

      1. creates the virtual environment, if it is not there yet
      2. installs llmorch into it
      3. puts that environment's Scripts folder on your user PATH, so `llmorch`
         works from any new PowerShell window, in any directory
      4. puts an "llmorch" shortcut on your Desktop that opens its own window

    Run it once:

        powershell -ExecutionPolicy Bypass -File .\setup.ps1

    After that, either double-click the Desktop icon or type `llmorch` anywhere.
#>

[CmdletBinding()]
param(
    [switch]$NoShortcut,
    [switch]$NoPath
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$scripts = Join-Path $venv 'Scripts'

Write-Host ''
Write-Host 'Setting up llmorch' -ForegroundColor Cyan
Write-Host "  folder: $root"

# --- 1. the virtual environment -------------------------------------------
if (-not (Test-Path $python)) {
    Write-Host '  creating .venv ...'
    $launcher = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' }
                elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' }
                else { $null }
    if (-not $launcher) {
        Write-Host ''
        Write-Host 'Python was not found.' -ForegroundColor Red
        Write-Host 'Install it from https://python.org/downloads (tick "Add python.exe to PATH"), then run this again.'
        exit 1
    }
    & $launcher -m venv $venv
    if (-not (Test-Path $python)) {
        Write-Host 'Could not create .venv' -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host '  .venv is already there'
}

# --- 2. install ------------------------------------------------------------
Write-Host '  installing llmorch ...'
& $python -m pip install --quiet --upgrade pip
& $python -m pip install --quiet -e "$root"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Install failed.' -ForegroundColor Red
    exit 1
}

# --- 3. PATH ---------------------------------------------------------------
# The reason `llmorch` is "not recognized": the executable exists, in a folder
# Windows was never told to look in.
if (-not $NoPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $userPath) { $userPath = '' }
    if ($userPath.Split(';') -notcontains $scripts) {
        [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $scripts).Trim(';'), 'User')
        Write-Host '  added to your PATH'
    } else {
        Write-Host '  already on your PATH'
    }
    # So it works in THIS window too, not only in new ones.
    $env:Path = $env:Path + ';' + $scripts
}

# --- 4. the Desktop shortcut ----------------------------------------------
if (-not $NoShortcut) {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $link = Join-Path $desktop 'llmorch.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($link)
    $shortcut.TargetPath = 'powershell.exe'
    $shortcut.Arguments = "-NoExit -ExecutionPolicy Bypass -Command `"& '$python' -m llmorch`""
    $shortcut.WorkingDirectory = $root
    $shortcut.IconLocation = 'powershell.exe,0'
    $shortcut.Description = 'Start an llmorch session'
    $shortcut.Save()
    Write-Host "  shortcut on your Desktop: $link"
}

# --- what to do next -------------------------------------------------------
Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
Write-Host '  Double-click "llmorch" on your Desktop, or type llmorch in a new PowerShell window.'
Write-Host ''

if (-not (Test-Path (Join-Path $root '.env'))) {
    Write-Host 'One thing left, and only you can do it:' -ForegroundColor Yellow
    Write-Host '  Without an API key llmorch replays a canned script and ignores what you type.'
    Write-Host '  Get a free key at https://console.groq.com, then put this in a file called .env here:'
    Write-Host ''
    Write-Host '      GROQ_API_KEY=your_key_here'
    Write-Host ''
    Write-Host '  Then start it with:  llmorch --live --providers groq'
    Write-Host ''
}
