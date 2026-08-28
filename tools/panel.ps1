# =============================================================================
#  A股流水线 本地控制台
#
#  为什么要有这个：两条流水线的触发依赖 GitHub cron 和本机计划任务，两者都
#  出过问题（cron 迟到 10 小时；机器睡着时任务错过时点）。出问题时最需要的是
#  一眼看清「今天到底跑没跑、跑到哪一步、有没有发信」，以及能立刻手动补一次。
#
#  纯只读 + 手动触发，不改任何配置。重复触发无害：脚本先查当天数据是否已产出，
#  workflow 那边还有串行组和幂等检查兜底。
#
#  用法：双击桌面的「A股流水线.cmd」，或 powershell -File tools\panel.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
$REPO = "SophiaSha2026/daily-report"
$PAGE = "https://sophiasha2026.github.io/daily-report/"

# 两条线的定义。加新线只要往这里加一项。
$LINES = @(
  @{ key="auction";  name="竞价选股"; wf="auction.yml";  prefix="auction";
     task="DailyReport-TriggerAuction";  due="09:27:30"; earliest="09:15"; page=$PAGE },
  @{ key="pullback"; name="形态扫描"; wf="pullback.yml"; prefix="pullback";
     task="DailyReport-TriggerPullback"; due="17:00 后"; earliest="17:00"; page=($PAGE + "pullback.html") }
)

function BJNow { (Get-Date).ToUniversalTime().AddHours(8) }

function Say($text, $color) {
  if ($color) { Write-Host $text -ForegroundColor $color } else { Write-Host $text }
}

function Line($ch) { Say ($ch * 78) DarkGray }

# --- gh 是否可用 -------------------------------------------------------------
function Check-Gh {
  $g = Get-Command gh -ErrorAction SilentlyContinue
  if (-not $g) {
    Say "找不到 gh 命令。请先安装 GitHub CLI：https://cli.github.com/" Red
    return $false
  }
  gh auth status *> $null
  if ($LASTEXITCODE -ne 0) {
    Say "gh 未登录。请运行：gh auth login" Red
    return $false
  }
  return $true
}

# --- 当天数据文件是否已产出 --------------------------------------------------
function Data-Exists($prefix, $bjDate) {
  $m = $bjDate.Substring(0,7)
  gh api "repos/$REPO/contents/data/$m/${prefix}_${bjDate}.parquet" --silent *> $null
  return ($LASTEXITCODE -eq 0)
}

# --- 取某条线今天的运行 ------------------------------------------------------
function Get-Runs($wf, $bjDate) {
  # 北京日期换成 UTC 起点：北京 00:00 = 前一天 16:00 UTC
  $since = ([datetime]::ParseExact($bjDate,"yyyy-MM-dd",$null)).AddHours(-8).ToString("yyyy-MM-ddTHH:mm:ssZ")
  $raw = gh run list --workflow=$wf --limit 20 `
         --json databaseId,createdAt,status,conclusion,event 2>$null
  if (-not $raw) { return @() }
  $all = $raw | ConvertFrom-Json
  return @($all | Where-Object { $_.createdAt -ge $since })
}

# --- 某次运行里发信那步的结论 ------------------------------------------------
function Send-Step($id) {
  $raw = gh run view $id --json jobs 2>$null
  if (-not $raw) { return "?" }
  $j = $raw | ConvertFrom-Json
  foreach ($job in $j.jobs) {
    foreach ($s in $job.steps) {
      if ($s.name -like "发信*") {
        if ($s.conclusion) { return $s.conclusion }
        return $s.status
      }
    }
  }
  return "-"
}

# --- 判定一条线今天的总状态 --------------------------------------------------
function Line-Status($line, $bjDate, $isTradingDay) {
  $r = [ordered]@{ label="未开始"; color="Yellow"; detail=""; runs=@() }
  if (-not $isTradingDay) {
    $r.label = "非交易日"; $r.color = "DarkGray"; $r.detail = "周末，无需运行"
    return $r
  }
  $runs = Get-Runs $line.wf $bjDate
  $r.runs = $runs
  $done  = Data-Exists $line.prefix $bjDate

  # 时点还没到，就不该按「今天没出结果」来判故障
  $hh, $mm = $line.earliest -split ":"
  $due = (BJNow).Date.AddHours([int]$hh).AddMinutes([int]$mm)
  $beforeDue = ((BJNow) -lt $due)

  $running = @($runs | Where-Object { $_.status -ne "completed" })
  $failed  = @($runs | Where-Object { $_.conclusion -in @("failure","cancelled","timed_out") })

  if ($running.Count -gt 0) {
    $r.label = "运行中"; $r.color = "Cyan"
    $r.detail = "$($running.Count) 个 job 进行中"
    return $r
  }
  if ($done) {
    # 数据已入库，再看有没有真的发出邮件
    $sent = $false
    foreach ($x in $runs) { if ((Send-Step $x.databaseId) -eq "success") { $sent = $true } }
    if ($sent) {
      $r.label = "成功"; $r.color = "Green"; $r.detail = "数据已入库，邮件已发出"
    } else {
      $r.label = "异常"; $r.color = "Magenta"
      $r.detail = "数据已入库，但没有一次运行成功发信 —— 检查 SMTP"
    }
    return $r
  }
  if ($failed.Count -gt 0) {
    $r.label = "失败"; $r.color = "Red"
    $r.detail = "$($failed.Count) 次运行失败，且当天数据未产出"
    return $r
  }
  if ($beforeDue) {
    $r.label = "未到时间"; $r.color = "DarkCyan"
    $r.detail = "应发时刻 $($line.due) 还没到（已有 $($runs.Count) 次运行结束，属正常的提前入口）"
    return $r
  }
  if ($runs.Count -gt 0) {
    $r.label = "异常"; $r.color = "Magenta"
    $r.detail = "$($runs.Count) 次运行都结束了，但当天数据没产出（多半是触发太早/太晚被守卫拦下）"
    return $r
  }
  $r.label = "未开始"; $r.color = "Red"
  $r.detail = "已过应发时刻，却一次运行都没有 —— 触发链全断了，立刻手动触发"
  return $r
}

# --- 本机计划任务 ------------------------------------------------------------
function Task-Info($name) {
  $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
  if (-not $t) { return @{ ok=$false; text="任务不存在" } }
  $i = Get-ScheduledTaskInfo -TaskName $name
  $bad = @()
  if ($t.Settings.DisallowStartIfOnBatteries) { $bad += "电池下不跑" }
  if ($t.State -eq "Disabled") { $bad += "已禁用" }
  $last = "从未运行"
  if ($i.LastRunTime -and $i.LastRunTime.Year -gt 2000) {
    $last = "{0} (结果 {1})" -f $i.LastRunTime, $i.LastTaskResult
  }
  $txt = "{0}  上次 {1}  下次 {2}" -f $t.State, $last, $i.NextRunTime
  return @{ ok=($bad.Count -eq 0); text=$txt; warn=($bad -join "，") }
}

# --- 主面板 ------------------------------------------------------------------
function Show-Panel {
  Clear-Host
  $bj = BJNow
  $bjDate = $bj.ToString("yyyy-MM-dd")
  $isTrade = ($bj.DayOfWeek -ne "Saturday" -and $bj.DayOfWeek -ne "Sunday")

  Line "="
  Say  "  A股流水线 控制台" White
  Say  ("  北京 " + $bj.ToString("yyyy-MM-dd HH:mm:ss ddd") +
        "     本机 " + (Get-Date).ToString("MM-dd HH:mm ddd")) Gray
  if (-not $isTrade) { Say "  今天是周末，两条线都不该运行" DarkGray }
  Line "="

  $script:STATUS = @{}
  foreach ($l in $LINES) {
    $st = Line-Status $l $bjDate $isTrade
    $script:STATUS[$l.key] = $st
    Say ""
    Say ("  【" + $l.name + "】 应发时刻 " + $l.due) White
    Write-Host "    状态: " -NoNewline
    Say $st.label $st.color
    Say ("    " + $st.detail) Gray
    if ($st.runs.Count -gt 0) {
      Say "    今日运行:" DarkGray
      foreach ($x in ($st.runs | Select-Object -First 4)) {
        $t = ([datetime]$x.createdAt).ToUniversalTime().AddHours(8).ToString("HH:mm")
        $c = $x.conclusion
        if (-not $c) { $c = $x.status }
        Say ("      北京 $t  $($x.event)  $c") DarkGray
      }
      if ($st.runs.Count -gt 4) { Say ("      ... 共 " + $st.runs.Count + " 次") DarkGray }
    }
    $ti = Task-Info $l.task
    Write-Host "    本机任务: " -NoNewline
    if ($ti.ok) { Say $ti.text Gray } else { Say ($ti.text + "  [" + $ti.warn + "]") Red }
  }

  Say ""
  Line "-"
  Say "  [1] 触发竞价    [2] 触发形态    [3] 刷新" White
  Say "  [4] 看最近一次运行的日志          [5] 打开在线面板" White
  Say "  [6] 本机任务体检（电源策略/唤醒定时器）   [0] 退出" White
  Line "-"
}

# --- 手动触发 ----------------------------------------------------------------
function Trigger($line) {
  $bjDate = (BJNow).ToString("yyyy-MM-dd")
  Say ""
  if (Data-Exists $line.prefix $bjDate) {
    Say ("  " + $line.name + " 今天（" + $bjDate + "）已经出过数据了。") Yellow
    $a = Read-Host "  仍然要再触发一次吗？(y/N)"
    if ($a -ne "y") { Say "  已取消" Gray; return }
  }
  Say ("  正在触发 " + $line.name + " ...") Cyan
  gh workflow run $line.wf --ref main -R $REPO
  if ($LASTEXITCODE -eq 0) {
    Say "  已派发。约 10 秒后可在面板看到「运行中」。" Green
  } else {
    Say "  派发失败，退出码 $LASTEXITCODE。检查网络和 gh 登录状态。" Red
  }
  Start-Sleep -Seconds 3
}

# --- 看日志 ------------------------------------------------------------------
function Show-Log {
  Say ""
  Say "  看哪条线的日志？ [1] 竞价  [2] 形态" White
  $k = Read-Host "  选择"
  $l = $LINES[0]; if ($k -eq "2") { $l = $LINES[1] }
  $raw = gh run list --workflow=$l.wf --limit 1 --json databaseId 2>$null
  if (-not $raw) { Say "  取不到运行记录" Red; Start-Sleep 2; return }
  $id = ($raw | ConvertFrom-Json)[0].databaseId
  Say ("  最近一次运行 $id 的关键日志：") Cyan
  Say ""
  $log = gh run view $id --log 2>$null
  if (-not $log) { Say "  日志还没生成（运行可能仍在进行）" Yellow }
  else {
    $log | Select-String -Pattern "INFO|WARNING|ERROR|已发送|形态匹配|初筛|入选|不扫描|抢救" |
      Select-Object -Last 25 | ForEach-Object {
        $s = $_.ToString() -replace "^[^\t]*\t[^\t]*\t", "" -replace "\x1b\[[0-9;]*m", ""
        Say ("    " + $s.Substring(0, [Math]::Min(120, $s.Length))) Gray
      }
  }
  Say ""
  Read-Host "  回车返回" | Out-Null
}

# --- 体检 --------------------------------------------------------------------
function Health {
  Say ""
  Say "  === 本机触发能力体检 ===" White
  $guid = ((powercfg /getactivescheme) -split " ")[3]
  $q = powercfg /query $guid SUB_SLEEP RTCWAKE
  $ac = (($q | Select-String "Current AC") -split ":")[1]
  $dc = (($q | Select-String "Current DC") -split ":")[1]
  $okWake = ($ac -match "0x00000001") -and ($dc -match "0x00000001")
  Write-Host "  唤醒定时器 (插电/电池): " -NoNewline
  if ($okWake) { Say "已启用 / 已启用  正常" Green }
  else { Say ("$ac / $dc   —— 未全部启用，机器睡着时任务叫不醒它") Red }

  foreach ($l in $LINES) {
    $t = Get-ScheduledTask -TaskName $l.task -ErrorAction SilentlyContinue
    Write-Host ("  " + $l.name + " 任务: ") -NoNewline
    if (-not $t) { Say "不存在" Red; continue }
    $rep = $t.Triggers | Where-Object { $_.Repetition.Interval } | Select-Object -First 1
    $hasLogon = @($t.Triggers | Where-Object { $_.CimClass.CimClassName -like "*Logon*" }).Count -gt 0
    $bits = @()
    $bits += "状态 $($t.State)"
    if ($rep) { $bits += "窗口内重试 $($rep.Repetition.Interval)/$($rep.Repetition.Duration)" }
    else { $bits += "无重试（只在单个时刻触发，机器睡着就错过）" }
    if ($hasLogon) { $bits += "开机登录触发 有" } else { $bits += "开机登录触发 无" }
    if (-not $t.Settings.DisallowStartIfOnBatteries) { $bits += "电池可跑" } else { $bits += "电池下不跑" }
    Say ($bits -join " | ") Gray
  }
  Say ""
  Say "  提醒：唤醒定时器只能叫醒**睡眠**中的机器。完全关机时任何本地方案都无效，" DarkGray
  Say "  那种情况只能靠 GitHub cron 和外部触发器。" DarkGray
  Say ""
  Read-Host "  回车返回" | Out-Null
}

# --- 主循环 ------------------------------------------------------------------
if (-not (Check-Gh)) { Read-Host "回车退出" | Out-Null; exit 1 }
while ($true) {
  Show-Panel
  $c = Read-Host "  选择"
  switch ($c) {
    "1" { Trigger $LINES[0] }
    "2" { Trigger $LINES[1] }
    "3" { }
    "4" { Show-Log }
    "5" { Start-Process $PAGE; Start-Process ($PAGE + "pullback.html") }
    "6" { Health }
    "0" { exit 0 }
    default { }
  }
}
