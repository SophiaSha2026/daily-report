# =============================================================================
#  A股流水线 控制台
#
#    1 跑全流程   选 早/晚 + 本地/远端，一路跑到邮件发出
#    2 面板       在线或本地
#    3 状态       流程状态 + 体检，一屏看完
#    4 日志       云端或本地
#    5 退出
#
#  远端 = 派发 GitHub Actions，带 LLM 文案，和每天自动跑的完全一样。
#         今天已经触发过（在跑或已出结果）时不重复派发，直接给状态。
#  本地 = 这台机器直接跑。实测本机可达全部行情源，连 runner 上不通的
#         新浪 vip 和东财都通，不排队。代价：没有 LLM 文案，发信要先配
#         tools\local.env。股票池两边一样大，本地优势是可用性不是覆盖面。
#
#  两条戒律：
#    · 本文件必须存成带 BOM 的 UTF-8。PowerShell 5.1 读脚本按系统 ANSI
#      码页，没有 BOM 中文会把字符串字面量拆坏。
#    · 不要用 PowerShell 的 here-string 语法内嵌脚本。需要内嵌就单独放
#      文件，比如体检那段在 tools\probe.py。
# =============================================================================
$ErrorActionPreference = "Continue"
$REPO = "SophiaSha2026/daily-report"
$ROOT = "C:\home\daily-report"
$PAGE = "https://sophiasha2026.github.io/daily-report/"
$PY   = "python"

$LINES = @(
  @{ key="auction";  name="竞价选股"; tag="早"; wf="auction.yml";  prefix="auction";
     task="DailyReport-TriggerAuction";  due="09:27:30"; earliest="09:15";
     out="out";          page=$PAGE },
  @{ key="pullback"; name="形态扫描"; tag="晚"; wf="pullback.yml"; prefix="pullback";
     task="DailyReport-TriggerPullback"; due="17:00";    earliest="17:00";
     out="out_pullback"; page=($PAGE + "pullback.html") }
)

function BJNow { (Get-Date).ToUniversalTime().AddHours(8) }
function Say($t, $c) { if ($c) { Write-Host $t -ForegroundColor $c } else { Write-Host $t } }
function Line($ch) { Say ($ch * 74) DarkGray }
function Pause { Say ""; Read-Host "  回车返回" | Out-Null }

function Pick-Line {
  Say ""
  Say "    [1] 竞价选股 [早]   北京 09:27:30 发信" Gray
  Say "    [2] 形态扫描 [晚]   北京 17:00 后扫描" Gray
  Say "    [0] 返回" DarkGray
  $k = Read-Host "  选哪条线"
  if ($k -eq "1") { return $LINES[0] }
  if ($k -eq "2") { return $LINES[1] }
  return $null
}

function Check-Gh {
  if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Say "找不到 gh。装 GitHub CLI: https://cli.github.com/" Red; return $false }
  gh auth status *> $null
  if ($LASTEXITCODE -ne 0) { Say "gh 未登录，先跑: gh auth login" Red; return $false }
  return $true
}

function Data-Exists($prefix, $bjDate) {
  $m = $bjDate.Substring(0,7)
  gh api "repos/$REPO/contents/data/$m/${prefix}_${bjDate}.parquet" --silent *> $null
  return ($LASTEXITCODE -eq 0)
}

function Get-Runs($wf, $bjDate) {
  $since = ([datetime]::ParseExact($bjDate,"yyyy-MM-dd",$null)).AddHours(-8).ToString("yyyy-MM-ddTHH:mm:ssZ")
  $raw = gh run list --workflow=$wf --limit 20 --json databaseId,createdAt,status,conclusion,event 2>$null
  if (-not $raw) { return @() }
  return @(($raw | ConvertFrom-Json) | Where-Object { $_.createdAt -ge $since })
}

function Send-Step($id) {
  $raw = gh run view $id --json jobs 2>$null
  if (-not $raw) { return "?" }
  foreach ($job in ($raw | ConvertFrom-Json).jobs) {
    foreach ($s in $job.steps) {
      if ($s.name -like "发信*") { if ($s.conclusion) { return $s.conclusion }; return $s.status }
    }
  }
  return "-"
}

function Local-Meta($line) {
  $f = Join-Path (Join-Path $ROOT $line.out) "run_meta.json"
  if (-not (Test-Path $f)) { return $null }
  try { return (Get-Content $f -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}

function Line-Status($line, $bjDate, $isTradingDay) {
  $r = [ordered]@{ label="未开始"; color="Yellow"; detail=""; runs=@() }
  if (-not $isTradingDay) {
    $r.label="非交易日"; $r.color="DarkGray"; $r.detail="周末，两条线都不该运行"; return $r }
  $runs = Get-Runs $line.wf $bjDate
  $r.runs = $runs
  $done = Data-Exists $line.prefix $bjDate
  $lm = Local-Meta $line
  $localToday = ($lm -and $lm.date -eq $bjDate)
  $hh, $mm = $line.earliest -split ":"
  $beforeDue = ((BJNow) -lt (BJNow).Date.AddHours([int]$hh).AddMinutes([int]$mm))
  $running = @($runs | Where-Object { $_.status -ne "completed" })
  $failed  = @($runs | Where-Object { $_.conclusion -in @("failure","cancelled","timed_out") })

  if ($running.Count -gt 0) {
    $r.label="运行中"; $r.color="Cyan"; $r.detail="$($running.Count) 个云端 job 进行中"; return $r }
  if ($done) {
    $sent = $false
    foreach ($x in $runs) { if ((Send-Step $x.databaseId) -eq "success") { $sent = $true } }
    if ($sent) { $r.label="成功"; $r.color="Green"; $r.detail="数据已入库，邮件已发出" }
    else { $r.label="异常"; $r.color="Magenta"; $r.detail="数据已入库，但没有一次云端运行成功发信" }
    return $r }
  if ($localToday) {
    $r.label="本地已出"; $r.color="Green"
    $r.detail="本地跑出了 $($lm.n) 只（云端无当日数据，属正常）"; return $r }
  if ($failed.Count -gt 0) {
    $r.label="失败"; $r.color="Red"; $r.detail="$($failed.Count) 次运行失败且无当日数据"; return $r }
  if ($beforeDue) {
    $r.label="未到时间"; $r.color="DarkCyan"
    $r.detail="应发时刻 $($line.due) 还没到（已有 $($runs.Count) 次提前入口结束，正常）"; return $r }
  if ($runs.Count -gt 0) {
    $r.label="异常"; $r.color="Magenta"
    $r.detail="$($runs.Count) 次运行都结束了但没产出（多半被守卫拦下，看日志）"; return $r }
  $r.label="未开始"; $r.color="Red"
  $r.detail="已过应发时刻却一次运行都没有 —— 触发链断了，用 [1] 手动跑"
  return $r
}

function Task-Info($name) {
  $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
  if (-not $t) { return @{ ok=$false; text="任务不存在"; warn="需要重建" } }
  $i = Get-ScheduledTaskInfo -TaskName $name
  $bad = @()
  if ($t.Settings.DisallowStartIfOnBatteries) { $bad += "电池下不跑" }
  if ($t.State -eq "Disabled") { $bad += "已禁用" }
  $last = "从未运行"
  if ($i.LastRunTime -and $i.LastRunTime.Year -gt 2000) {
    $last = "{0} (结果 {1})" -f $i.LastRunTime, $i.LastTaskResult }
  return @{ ok=($bad.Count -eq 0)
            text=("{0} | 上次 {1} | 下次 {2}" -f $t.State, $last, $i.NextRunTime)
            warn=($bad -join "，") }
}

function Watch-Run($id) {
  Say ("  盯 run " + $id + " 到跑完（竞价那条可能要等到 09:27:30）") Cyan
  Say "  Ctrl+C 可随时退出等待，云端会继续跑。" DarkGray
  Say ""
  $lastStep = ""
  while ($true) {
    $j = gh run view $id --json status,conclusion,jobs 2>$null
    if (-not $j) { Start-Sleep 15; continue }
    $o = $j | ConvertFrom-Json
    $cur = ""
    foreach ($job in $o.jobs) {
      foreach ($s in $job.steps) { if ($s.status -eq "in_progress") { $cur = $s.name } }
    }
    if ($cur -and $cur -ne $lastStep) {
      Say ("    " + (Get-Date -Format "HH:mm:ss") + "  " + $cur) Gray
      $lastStep = $cur
    }
    if ($o.status -eq "completed") {
      Say ""
      if ($o.conclusion -eq "success") { Say "  云端运行完成：success" Green }
      else { Say ("  云端运行结束：" + $o.conclusion) Red }
      $sent = Send-Step $id
      if ($sent -eq "success") { Say "  发信步骤：success —— 邮件已发出" Green }
      elseif ($sent -eq "skipped") { Say "  发信步骤：skipped —— 本轮没发信（被幂等或守卫跳过）" Yellow }
      else { Say ("  发信步骤：" + $sent) Yellow }
      break
    }
    Start-Sleep 15
  }
}

# --- 1 跑全流程 --------------------------------------------------------------
function Run-Flow {
  $line = Pick-Line
  if (-not $line) { return }
  Say ""
  Say ("  " + $line.name + " [" + $line.tag + "] —— 在哪跑？") White
  Say "    [1] 远端  GitHub Actions，带 LLM 文案，和每天自动跑的一样" Gray
  Say "    [2] 本地  这台机器直接跑，不排队；无 LLM 文案" Gray
  Say "    [0] 返回" DarkGray
  $w = Read-Host "  选择"
  if ($w -eq "1") { Run-Cloud $line }
  elseif ($w -eq "2") { Run-Local $line }
}

function Run-Cloud($line) {
  $bjDate = (BJNow).ToString("yyyy-MM-dd")
  Say ""
  Say "  先查今天远端跑过没有..." DarkGray
  $runs = Get-Runs $line.wf $bjDate
  $running = @($runs | Where-Object { $_.status -ne "completed" })
  $done = Data-Exists $line.prefix $bjDate

  # 已经在跑：不重复派发，直接接上去盯
  if ($running.Count -gt 0) {
    Say ""
    Say ("  今天远端已经在跑了（" + $running.Count + " 个 job 进行中），不重复派发。") Yellow
    Say ""
    Watch-Run $running[0].databaseId
    Pause; return
  }

  # 已经出过结果：直接显示状态，要重跑得明确确认
  if ($done) {
    Say ""
    Say ("  今天（" + $bjDate + "）远端已经跑过并出了数据，不重复派发。") Yellow
    Say ""
    $st = Line-Status $line $bjDate $true
    Write-Host "    状态: " -NoNewline; Say $st.label $st.color
    Say ("    " + $st.detail) Gray
    foreach ($x in ($runs | Select-Object -First 5)) {
      $t = ([datetime]$x.createdAt).ToUniversalTime().AddHours(8).ToString("HH:mm")
      $c = $x.conclusion; if (-not $c) { $c = $x.status }
      Say ("      北京 " + $t + "  " + $x.event + "  " + $c +
           "  发信=" + (Send-Step $x.databaseId)) DarkGray
    }
    Say ""
    $a = Read-Host "  确实要强制重跑一次吗？(y/N)"
    if ($a -ne "y") { Say "  已取消" Gray; Pause; return }
    $extra = @("-f","force=true")
  } else {
    $extra = @()
  }

  if ($line.key -eq "auction") {
    $bj = BJNow
    if ($bj.Hour -gt 9 -or ($bj.Hour -eq 9 -and $bj.Minute -ge 27)) {
      Say ""
      Say "  09:25-09:30 竞价窗口已过。" Yellow
      Say "  抢救模式：竞价价用今开（精确），量能维度停用、斜率/形态维度失效。" Yellow
      $a = Read-Host "  用抢救模式？(Y/n)"
      if ($a -ne "n") { $extra += @("-f","late=true") }
    }
  }
  Say ""
  Say "  派发中..." Cyan
  gh workflow run $line.wf --ref main -R $REPO @extra
  if ($LASTEXITCODE -ne 0) { Say "  派发失败（退出码 $LASTEXITCODE）" Red; Pause; return }
  Start-Sleep 8
  $raw = gh run list --workflow=$line.wf --limit 1 --json databaseId 2>$null
  if (-not $raw) { Say "  已派发，但取不到 run id。去 [3] 状态里看。" Yellow; Pause; return }
  Say ""
  Watch-Run ($raw | ConvertFrom-Json)[0].databaseId
  Pause
}

function Load-Env {
  $f = Join-Path (Join-Path $ROOT "tools") "local.env"
  $map = @{}
  if (Test-Path $f) {
    foreach ($l in (Get-Content $f -Encoding UTF8)) {
      if ($l -match '^\s*#') { continue }
      if ($l -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') { $map[$Matches[1]] = $Matches[2] }
    }
  }
  return $map
}

function Run-Py($argList, $envmap, $tag) {
  $old = @{}
  foreach ($k in $envmap.Keys) {
    $old[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $envmap[$k])
  }
  $env:PYTHONIOENCODING = "utf-8"
  # 早晚两条线各写各的日志，混在一起排查时分不清谁是谁
  $logf = Join-Path (Join-Path $ROOT "tools") ("local_run_" + $tag + ".log")
  Push-Location $ROOT
  & $PY @argList 2>&1 | ForEach-Object {
    $s = "$_"
    Say ("    " + $s) DarkGray
    Add-Content -Path $logf -Value ((Get-Date -Format "HH:mm:ss") + " " + $s) -Encoding UTF8
  }
  $code = $LASTEXITCODE
  Pop-Location
  foreach ($k in $old.Keys) { [Environment]::SetEnvironmentVariable($k, $old[$k]) }
  return $code
}

function Run-Local($line) {
  Say ""
  $envmap = Load-Env
  $canMail = ($envmap.ContainsKey("SMTP_HOST") -and $envmap.ContainsKey("MAIL_TO"))
  if ($canMail) { Say ("  发信配置已就绪，跑完发到 " + $envmap["MAIL_TO"]) Green }
  else {
    Say "  没配 tools\local.env —— 跑完只出本地面板，不会发邮件。" Yellow
    Say "  想本地也发信：复制 tools\local.env.example 成 local.env 填好，" DarkGray
    Say "  里面几个值和你 GitHub Secrets 里的一模一样。" DarkGray
  }
  Say "  本地没有 LLM 文案（那步要 claude-code-action），只有量化结果。" DarkGray
  $a = Read-Host "  开始？(Y/n)"
  if ($a -eq "n") { return }
  Say ""
  $t0 = Get-Date

  if ($line.key -eq "auction") {
    $meta = Join-Path (Join-Path $ROOT "cache") "universe_meta.json"
    $need = $true
    if (Test-Path $meta) {
      try {
        $j = Get-Content $meta -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($j.date -eq (BJNow).ToString("yyyy-MM-dd")) { $need = $false }
      } catch { }
    }
    if ($need) {
      Say "  [1/3] 候选池不是今天的，先建（3-5 分钟）..." Cyan
      Run-Py @("src\premarket.py") $envmap $line.key | Out-Null
    } else { Say "  [1/3] 候选池已是今天的，跳过" Gray }
    $bj = BJNow
    $lateArg = @()
    if ($bj.Hour -gt 9 -or ($bj.Hour -eq 9 -and $bj.Minute -ge 27)) {
      Say "  竞价窗口已过，用抢救模式（今开当竞价价，量能维度停用）" Yellow
      $lateArg = @("--late")
    }
    Say "  [2/3] 采集 + 打分..." Cyan
    $c = Run-Py (@("src\run_auction.py","--stage","quick") + $lateArg) $envmap $line.key
    if ($c -ne 0) { Say "  采集失败，退出码 $c" Red; Pause; return }
    Say "  [3/3] 生成面板 + 附件 + 发信..." Cyan
    Run-Py @("src\run_auction.py","--stage","enrich") $envmap $line.key | Out-Null
  } else {
    Say "  [1/2] 收盘后扫描..." Cyan
    $c = Run-Py @("src\pullback.py","--stage","scan") $envmap $line.key
    if ($c -ne 0) { Say "  扫描失败，退出码 $c" Red; Pause; return }
    Say "  [2/2] 生成面板 + 附件 + 发信..." Cyan
    Run-Py @("src\pullback.py","--stage","send") $envmap $line.key | Out-Null
  }

  Say ""
  Say ("  完成，耗时 " + [int]((Get-Date) - $t0).TotalSeconds + " 秒") Green
  $lm = Local-Meta $line
  if ($lm) { Say ("  本地产物：" + $lm.date + "，" + $lm.n + " 只") Green }
  if ($canMail) { Say "  邮件已发出（上面日志里有「已发送」那行）" Green }
  else { Say "  没发邮件（未配 local.env）。用 [2] 面板看本地结果。" Yellow }
  $p = Join-Path (Join-Path $ROOT $line.out) "panel.html"
  if (Test-Path $p) {
    $a = Read-Host "  打开本地面板？(Y/n)"
    if ($a -ne "n") { Start-Process $p }
  }
  Pause
}

# --- 2 面板 ------------------------------------------------------------------
function Open-Panel {
  $line = Pick-Line
  if (-not $line) { return }
  Say ""
  Say "    [1] 在线（GitHub Pages）   [2] 本地产物" Gray
  $k = Read-Host "  选择"
  if ($k -eq "1") { Start-Process $line.page; return }
  if ($k -eq "2") {
    $p = Join-Path (Join-Path $ROOT $line.out) "panel.html"
    if (Test-Path $p) { Start-Process $p }
    else { Say "  这条线还没有本地产物，先用 [1] 里的「本地」跑一次" Yellow; Start-Sleep 2 }
  }
}

# --- 3 状态（流程 + 体检）----------------------------------------------------
function Show-Status {
  Clear-Host
  $bj = BJNow
  $bjDate = $bj.ToString("yyyy-MM-dd")
  $isTrade = ($bj.DayOfWeek -ne "Saturday" -and $bj.DayOfWeek -ne "Sunday")
  Line "="
  Say ("  状态总览    北京 " + $bj.ToString("yyyy-MM-dd HH:mm ddd") +
       "    本机 " + (Get-Date).ToString("MM-dd HH:mm")) White
  Line "="

  foreach ($l in $LINES) {
    $st = Line-Status $l $bjDate $isTrade
    Say ""
    Say ("  【" + $l.name + " " + $l.tag + "】 应发 " + $l.due) White
    Write-Host "    流程: " -NoNewline; Say $st.label $st.color
    Say ("          " + $st.detail) Gray
    foreach ($x in ($st.runs | Select-Object -First 3)) {
      $t = ([datetime]$x.createdAt).ToUniversalTime().AddHours(8).ToString("HH:mm")
      $c = $x.conclusion; if (-not $c) { $c = $x.status }
      Say ("          云端 " + $t + "  " + $x.event + "  " + $c) DarkGray
    }
    $lm = Local-Meta $l
    if ($lm) { Say ("          本地产物 " + $lm.date + "  " + $lm.n + " 只") DarkGray }
    $ti = Task-Info $l.task
    Write-Host "    任务: " -NoNewline
    if ($ti.ok) { Say $ti.text Gray } else { Say ($ti.text + "  [" + $ti.warn + "]") Red }
  }

  Say ""; Line "-"; Say "  体检" White

  $guid = ((powercfg /getactivescheme) -split " ")[3]
  $q = powercfg /query $guid SUB_SLEEP RTCWAKE
  $ac = (($q | Select-String "Current AC") -split ":")[1]
  $dc = (($q | Select-String "Current DC") -split ":")[1]
  Write-Host "    唤醒定时器 插电/电池: " -NoNewline
  if (($ac -match "0x00000001") -and ($dc -match "0x00000001")) { Say "启用 / 启用  正常" Green }
  else { Say ("$ac / $dc  未全部启用，机器睡着时叫不醒") Red }

  Say "    行情源与依赖（东财是可选源，FAIL 不影响出榜）:" Gray
  $env:PYTHONIOENCODING = "utf-8"
  & $PY (Join-Path (Join-Path $ROOT "tools") "probe.py") 2>&1 | ForEach-Object {
    if ("$_" -match "OK") { Say "$_" Green } else { Say "$_" Red } }

  $e = Load-Env
  Write-Host "    本地发信: " -NoNewline
  if ($e.ContainsKey("SMTP_HOST") -and $e.ContainsKey("MAIL_TO")) {
    Say ("已配置，收件人 " + $e["MAIL_TO"]) Green }
  else { Say "未配置 tools\local.env —— 本地跑只出面板不发邮件" Yellow }

  Say ""
  Say "  触发层：Cloudflare(北京07:30/17:05) + 本机任务(窗口内重试+开机触发)" DarkGray
  Say "  唤醒定时器只能叫醒睡眠中的机器；完全关机时靠 Cloudflare 那层。" DarkGray
  Pause
}

# --- 4 日志 ------------------------------------------------------------------
function Show-Log {
  $line = Pick-Line
  if (-not $line) { return }
  Say ""
  Say "    [1] 云端最近一次   [2] 本地最近一次" Gray
  $k = Read-Host "  选择"
  if ($k -eq "2") {
    $f = Join-Path (Join-Path $ROOT "tools") ("local_run_" + $line.key + ".log")
    if (Test-Path $f) { Get-Content $f -Tail 45 -Encoding UTF8 | ForEach-Object { Say ("    " + $_) Gray } }
    else { Say "  这条线还没有本地运行记录" Yellow }
    Pause; return
  }
  if ($k -ne "1") { return }
  $raw = gh run list --workflow=$line.wf --limit 1 --json databaseId 2>$null
  if (-not $raw) { Say "  取不到运行记录" Red; Pause; return }
  $id = ($raw | ConvertFrom-Json)[0].databaseId
  Say ("  云端 run " + $id + " ：") Cyan
  $log = gh run view $id --log 2>$null
  if (-not $log) { Say "  日志未生成（可能仍在运行）" Yellow }
  else {
    $log | Select-String -Pattern "INFO|WARNING|ERROR|已发送|形态匹配|初筛|入选|不扫描|抢救" |
      Select-Object -Last 30 | ForEach-Object {
        $s = $_.ToString() -replace "^[^\t]*\t[^\t]*\t","" -replace "\x1b\[[0-9;]*m",""
        Say ("    " + $s.Substring(0,[Math]::Min(116,$s.Length))) Gray }
  }
  Pause
}

# --- 主菜单 ------------------------------------------------------------------
function Menu {
  Clear-Host
  $bj = BJNow
  Line "="
  Say "  A股流水线 控制台" White
  Say ("  北京 " + $bj.ToString("yyyy-MM-dd HH:mm:ss ddd") +
       "     本机 " + (Get-Date).ToString("MM-dd HH:mm")) Gray
  Line "="
  Say ""
  Say "    [1] 跑全流程    选 早/晚 + 本地/远端，一路跑到邮件发出" White
  Say "    [2] 面板        在线 或 本地" White
  Say "    [3] 状态        流程状态 + 体检，一屏看完" White
  Say "    [4] 日志        云端 或 本地" White
  Say "    [5] 退出" White
  Say ""
  Line "-"
}

if (-not (Check-Gh)) { Read-Host "回车退出" | Out-Null; exit 1 }
while ($true) {
  Menu
  $c = Read-Host "  选择"
  switch ($c) {
    "1" { Run-Flow }
    "2" { Open-Panel }
    "3" { Show-Status }
    "4" { Show-Log }
    "5" { exit 0 }
    "0" { exit 0 }
    default { }
  }
}
