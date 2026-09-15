<#
本机计划任务的唯一定义。改排期改这里，然后重跑：

    powershell -ExecutionPolicy Bypass -File tools\setup_tasks.ps1
    powershell -ExecutionPolicy Bypass -File tools\setup_tasks.ps1 -Only Evening,Sync

四个任务都经 headless.vbs 起（无黑框），都带「登录时触发」，都允许唤醒、
允许电池、错过就补跑（StartWhenAvailable）、已在跑就忽略新触发。
入口是 run_local.cmd -> local_run.py --if-needed：跑过就不再跑，
不在窗口、或还没到自动开跑时刻（等手动），就只拉一次远端。
所以反复触发是安全的，这正是设计：机器可能在任何一个时刻睡着，
只要在窗口内醒过一次就行。优先级：手动 > 本机自动 > 云端。

时间全是**本机（美东）**时间，Windows 会跟着夏令时走。对应的北京窗口在
src/local_run.py 的 FLOWS 里判，两边要一起看：
    Morning  周日~周四 18:00 起每 15 分钟，共 3h15m   = 北京 06:00~09:15（冬令时 07:00~10:15，窗口 09:16 关）
    Evening  周一~周五 04:30 起每 30 分钟，共 16h     = 北京 16:30~次日 08:30
    Learn    周一~周五 04:40 起每 30 分钟，共 16h
    Sync     每天 00:15 起每 30 分钟，整天             只拉远端，几秒钟
#>
param(
    [string[]]$Only = @("Morning", "Evening", "Learn", "Sync")
)
$ErrorActionPreference = "Stop"
# powershell -File 传数组参数时整个当一个字符串给进来，自己按逗号拆
$Only = @($Only | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
$vbs = Join-Path $root "tools\headless.vbs"
$cmd = Join-Path $root "tools\run_local.cmd"
$user = "$env:USERDOMAIN\$env:USERNAME"

function Rep([string]$every, [string]$for) {
    # 用 -Once 造一个带重复的触发器，把它的 Repetition 拷给周触发器
    $t = New-ScheduledTaskTrigger -Once -At "00:00" `
        -RepetitionInterval ([System.Xml.XmlConvert]::ToTimeSpan($every)) `
        -RepetitionDuration ([System.Xml.XmlConvert]::ToTimeSpan($for))
    return $t.Repetition
}

function Weekly([string[]]$days, [string]$at, [string]$every, [string]$for) {
    $t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $at
    $t.Repetition = Rep $every $for
    return $t
}

function Daily([string]$at, [string]$every, [string]$for) {
    $t = New-ScheduledTaskTrigger -Daily -At $at
    $t.Repetition = Rep $every $for
    return $t
}

$defs = @{
    Morning = @{
        Flow = "morning"; Limit = "PT4H"
        Trigger = Weekly @("Sunday","Monday","Tuesday","Wednesday","Thursday") "18:00" "PT15M" "PT3H15M"
        Desc = "早盘选股（竞价线）。北京 09:27:30 发信；本机为主，云端只在本机没发时代发。"
    }
    Evening = @{
        Flow = "breakout"; Limit = "PT3H"
        Trigger = Weekly @("Monday","Tuesday","Wednesday","Thursday","Friday") "04:30" "PT30M" "PT16H"
        Desc = "起涨预测（晚间系统）。目标日是最近一个已收盘交易日，北京 16:00 到次日 08:30 都能补跑。"
    }
    Learn = @{
        Flow = "learn"; Limit = "PT3H"
        Trigger = Weekly @("Monday","Tuesday","Wednesday","Thursday","Friday") "04:40" "PT30M" "PT16H"
        Desc = "参数自学（学习线）。收盘后跑，窗口同起涨预测。"
    }
    Sync = @{
        Flow = "sync"; Limit = "PT10M"
        Trigger = Daily "00:15" "PT30M" "P1D"
        Desc = "同步远端产物。云端替本机跑出来的清单拉到本地面板，几秒钟。"
    }
}

foreach ($name in $Only) {
    $d = $defs[$name]
    if (-not $d) { Write-Host "未知任务 $name"; continue }
    $task = "DailyReport-Local-$name"
    $action = New-ScheduledTaskAction -Execute "wscript.exe" `
        -Argument "//B //Nologo `"$vbs`" `"$cmd`" `"$($d.Flow)`"" -WorkingDirectory $root
    $logon = New-ScheduledTaskTrigger -AtLogOn -User $user
    $settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit ([System.Xml.XmlConvert]::ToTimeSpan($d.Limit))
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $task -Action $action -Trigger @($d.Trigger, $logon) `
        -Settings $settings -Principal $principal -Description $d.Desc -Force | Out-Null
    $i = Get-ScheduledTaskInfo -TaskName $task
    Write-Host ("{0,-28} 已注册  下次 {1}" -f $task, $i.NextRunTime)
}
