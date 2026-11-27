# Render one still of a scape and develop it to a JPG.
#   .\render.ps1 -Scape 01_iwagumi -Frame 800 -Quality draft -Name iwagumi_a [-Crop "x,y,w,h"] [-Extra "--lamp-watts 200"]
param(
    [string]$Scape = "01_iwagumi",
    [int]$Frame = 800,
    [string]$Quality = "draft",
    [string]$Name = "test",
    [string]$Extra = ""
)
# Blender writes warnings to stderr; they must not abort the script, so its
# output is captured by cmd rather than by PowerShell.
$ErrorActionPreference = "Continue"
$work = if ($env:AQUARIUM_WORK_DIR) { $env:AQUARIUM_WORK_DIR } else { "$env:LOCALAPPDATA\aquarium_dream" }
$dir = "$work\renders\tests"
New-Item -ItemType Directory -Force $dir | Out-Null
$blender = "$env:ProgramFiles\Blender Foundation\Blender 4.5\blender.exe"
$log = "$dir\$Name.log"
$extraArgs = if ($Extra) { $Extra -split ' ' } else { @() }
cmd /c "`"$blender`" -b --python `"$PSScriptRoot\src\blender\build.py`" -- --scape $Scape --frame $Frame --quality $Quality --out `"$dir\$Name.exr`" $Extra > `"$log`" 2>&1"
Select-String -Path $log -Pattern '^\[build\]|^\[render\]|Error|Traceback|^\s+File .*line \d+|Exception' | ForEach-Object { $_.Line } | Select-Object -First 25
python "$PSScriptRoot\src\comp\develop.py" "$dir\$Name.exr" "$dir\$Name.jpg"
