param([switch]$Fast)

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir

$gpp = Get-Command g++ -ErrorAction SilentlyContinue
if (-not $gpp) {
    $candidates = @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_*\mingw64\bin\g++.exe",
        "C:\mingw64\bin\g++.exe"
    )
    foreach ($pat in $candidates) {
        $found = Get-ChildItem $pat -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { $gpp = $found.FullName; break }
    }
}
if (-not $gpp) { throw "g++ not found. Install WinLibs mingw or add g++ to PATH." }

$flags = @("-O3", "-funroll-loops", "-static", "-static-libgcc", "-static-libstdc++")
if ($Fast) {
    $flags += "-DTIMEOUT_USEC=100000", "-DFIRSTMOVE_USEC=200000", "-DVERBOSE=0"
}

& $gpp @flags -o MyTronBot.exe MyTronBot.cc
if ($LASTEXITCODE -ne 0) { throw "MyTronBot build failed" }
Write-Host "Built $dir\MyTronBot.exe$(if ($Fast) { ' (fast timers)' })"
