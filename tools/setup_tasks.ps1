<#
本机计划任务的唯一定义。改排期改这里，然后重跑：

    powershell -ExecutionPolicy Bypass -File tools\setup_tasks.ps1
    powershell -ExecutionPolicy Bypass -File tools\setup_tasks.ps1 -Only Pullback,Sync

三个任务都经 headless.vbs 起（无黑框），都带「登录时触发」，都允许唤醒、
允许电池、错过就补跑（StartWhenAvailable）、已在跑就忽略新触发。
入口是 run_local.cmd -> local_run.py --if-needed：跑过就不再跑，
不在窗口、或还没到自动开跑时刻（等手动），就只拉一次远端。
所以反复触发是安全的，这正是设计：机器可能在任何一个时刻睡着，
只要在窗口内醒过一次就行。优先级：手动 > 本机自动 > 云端。

时刻一律写 **UTC**，不写本机时刻（2026-09-16 改）
------------------------------------------------
New-ScheduledTaskTrigger -At 收的是本机时刻，但它会把**注册那一刻**的 UTC
偏移烘进 StartBoundary：实测导出四个已注册任务，全是
2026-09-12T18:00:00-04:00 这种带偏移的形式。微软 ITrigger::put_StartBoundary
原文：「When an offset is specified (using hours and minutes or Z), then the
time and offset are always used regardless of the time zone and daylight
saving settings on the local computer.」

所以裸 -At 的真实后果**不是**「Windows 跟着夏令时走」，正好相反：触发器是
绝对时刻，夏天注册和冬天注册会差一小时**北京**时间。漂的是美东墙钟，
北京时刻不漂。而这条流水线所有的开跑窗口（src/local_run.py 的 FLOWS）判的
都是北京时间，所以这里直接按 UTC 写死，注册季节就影响不到排期。
对应关系（UTC + 8 = 北京）：

    Evening   08:30Z 周一~周五 起每 30 分钟，共 16h   = 北京 16:30~次日 08:00（起涨预测）
    Pullback  09:40Z 周一~周五 起每 15 分钟，共 15h   = 北京 17:40~次日 08:25（长期调整突破，17:58 发信）
    Sync      04:15Z 每天 起每 30 分钟，整天          = 北京 12:15 起，只拉远端，几秒钟

DaysOfWeek 仍按**本机**日期数：08:30Z / 09:40Z 在美东是同一天凌晨
（EDT 04:30 / 05:40），北京日期和美东日期是同一天，所以写 Monday~Friday。
改时刻前先把这个换算算一遍（早盘那条 22:00Z 就是美东前一天傍晚，要写
Sunday~Thursday）。改完重跑脚本，用 Export-ScheduledTask 核对 StartBoundary：
它可能显示成 -04:00/-05:00，换算到 UTC 等于上表就对。

2026-09-27 早盘系统归档：Morning（早盘选股）和 Learn（参数自学）两个任务从这里
删掉，本机上已注册的那两个用 Unregister-ScheduledTask 删了；定义原样留在
archive/morning/tools/setup_tasks.ps1。
src/selftest_gui.py::check_task_schedule 会按上表复算北京敲击序列，
断言每条线都至少敲中一次「自动开跑时刻」。
#>
param(
    [string[]]$Only = @("Evening", "Pullback", "Sync")
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

function Utc([string]$hm) {
    # 直接写 UTC（带 Z）。-At 只能给本机时刻，而注册季节会被烘死在
    # StartBoundary 里（见文件头），北京时刻就跟着重跑脚本的季节漂。
    # 日期用今天：周/日触发器只拿 StartBoundary 的日期当起算点。
    return (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd") + "T" + $hm + ":00Z"
}

function Weekly([string[]]$days, [string]$atUtc, [string]$every, [string]$for) {
    $t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At "00:00"
    $t.Repetition = Rep $every $for
    $t.StartBoundary = (Utc $atUtc)
    return $t
}

function Daily([string]$atUtc, [string]$every, [string]$for) {
    $t = New-ScheduledTaskTrigger -Daily -At "00:00"
    $t.Repetition = Rep $every $for
    $t.StartBoundary = (Utc $atUtc)
    return $t
}

# 第二个参数是 **UTC** 时刻（+8 就是北京），不是本机时刻。理由见文件头。
$defs = @{
    Evening = @{
        Flow = "breakout"; Limit = "PT3H"
        Trigger = Weekly @("Monday","Tuesday","Wednesday","Thursday","Friday") "08:30" "PT30M" "PT16H"
        Desc = "起涨预测（晚间系统）。目标日是最近一个已收盘交易日，北京 16:00 到次日 08:30 都能补跑。"
    }
    Pullback = @{
        Flow = "pullback"; Limit = "PT4H"
        Trigger = Weekly @("Monday","Tuesday","Wednesday","Thursday","Friday") "09:40" "PT15M" "PT15H"
        Desc = "长期调整突破（晚间系统）。北京 17:40 起跑、17:58 发信；目标日是最近一个已收盘交易日，到次日 08:30 都能补。"
    }
    Sync = @{
        Flow = "sync"; Limit = "PT10M"
        Trigger = Daily "04:15" "PT30M" "P1D"
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
    # 「登录时触发」是同一瞬间：2026-09-15 和 09-16 两次开机，起涨预测和
    # 学习线同一秒启动，一起 pull、一起写 git 索引。长期调整突破的登录触发
    # 延后 5 分钟错开，让起涨预测先把日线补上（排期触发不动）。
    if ($name -eq "Pullback") { $logon.Delay = "PT5M" }
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
