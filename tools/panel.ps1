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
$ROOT = "C:\home\daily-report"
$PY   = "python"

# 两条线的定义。加新线只要往这里加一项。
$LINES = @(
  @{ key="auction";  name="竞价选股"; wf="auction.yml";  prefix="auction";
     task="DailyReport-TriggerAuction";  due="09:27:30"; earliest="09:15"; out="out"; page=$PAGE },
  @{ key="pullback"; name="形态扫描"; wf="pullback.yml"; prefix="pullback";
     task="DailyReport-TriggerPullback"; due="17:00 后"; earliest="17:00"; out="out_pullback"; page=($PAGE + "pullback.html") }
)

function BJNow { (Get-Date).ToUniversalTime().AddHours(8) }

function Say($text, $color) {
  if ($color) { Write-Host $text -ForegroundColor $color } else { Write-Host $text }
}

function Line($ch) { Say ($ch * 78) DarkGray }
function Pause { Say ""; Read-Host "  回车返回" | Out-Null }

# 本地跑完也会写 run_meta.json，所以云端和本地两种产物面板都看得到
function Local-Meta($line) {
  $f = Join-Path (Join-Path $ROOT $line.out) "run_meta.json"
  if (-not (Test-Path $f)) { return $null }
  try { return (Get-Content $f -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}

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
  $lm = Local-Meta $line
  $localToday = ($lm -and $lm.date -eq $bjDate)

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
  if ($localToday) {
    $r.label = "本地已出"; $r.color = "Green"
    $r.detail = "本地跑出了 $($lm.n) 只（云端无当日数据文件，属正常）"
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
    $lm2 = Local-Meta $l
    if ($lm2) { Say ("      本地产物 " + $lm2.date + "  " + $lm2.n + " 只") DarkGray }
    $ti = Task-Info $l.task
    Write-Host "    本机任务: " -NoNewline
    if ($ti.ok) { Say $ti.text Gray } else { Say ($ti.text + "  [" + $ti.warn + "]") Red }
  }

  Say ""
  Line "-"
  Say "  [1] 触发竞价      [2] 触发形态      每项可选 云端 / 本地完整跑" White
  Say "  [3] 刷新          [4] 看日志        [5] 打开面板（在线/本地）" White
  Say "  [6] 体检（电源·任务·行情源·依赖·发信配置）        [0] 退出" White
  Line "-"
}

# --- 触发 --------------------------------------------------------------------
function Trigger($line) {
  Say ""
  Say ("  触发 " + $line.name + " —— 选哪种：") White
  Say "    [1] 云端 GitHub Actions   和平时自动跑的一样，带 LLM 分析文案" Gray
  Say "    [2] 本地完整跑            不排队、不看 GitHub 脸色；无 LLM 文案" Gray
  Say "    [0] 返回" Gray
  $m = Read-Host "  选择"
  if ($m -eq "1") { Trigger-Cloud $line }
  elseif ($m -eq "2") { Trigger-Local $line }
}

function Trigger-Cloud($line) {
  $bjDate = (BJNow).ToString("yyyy-MM-dd")
  $force = "false"
  if (Data-Exists $line.prefix $bjDate) {
    Say ("  云端今天（" + $bjDate + "）已有数据。") Yellow
    $a = Read-Host "  强制重跑？(y/N)"
    if ($a -ne "y") { Say "  已取消" Gray; Start-Sleep 1; return }
    $force = "true"
  }
  $extra = @("-f", "force=$force")
  if ($line.key -eq "auction") {
    $bj = BJNow
    if ($bj.Hour -gt 9 -or ($bj.Hour -eq 9 -and $bj.Minute -ge 27)) {
      Say "  09:25-09:30 竞价窗口已过。" Yellow
      Say "  抢救模式：竞价价用今开（精确），但量能维度停用、斜率/形态维度失效。" Yellow
      $a = Read-Host "  用抢救模式？(Y/n)"
      if ($a -ne "n") { $extra += @("-f", "late=true") }
    }
  }
  Say "  派发中..." Cyan
  gh workflow run $line.wf --ref main -R $REPO @extra
  if ($LASTEXITCODE -eq 0) { Say "  已派发。约 10 秒后面板显示「运行中」。" Green }
  else { Say "  派发失败（退出码 $LASTEXITCODE）" Red }
  Start-Sleep 3
}

function Load-Env {
  # tools\local.env：本地发信用的 SMTP 配置，已在 .gitignore 排除
  $f = Join-Path $ROOT "tools\local.env"
  $map = @{}
  if (Test-Path $f) {
    foreach ($l in (Get-Content $f -Encoding UTF8)) {
      if ($l -match '^\s*#') { continue }
      if ($l -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') { $map[$Matches[1]] = $Matches[2] }
    }
  }
  return $map
}

function Run-Py($argList, $envmap) {
  $old = @{}
  foreach ($k in $envmap.Keys) {
    $old[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $envmap[$k])
  }
  $env:PYTHONIOENCODING = "utf-8"
  $logf = Join-Path $ROOT "tools\local_run.log"
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

function Trigger-Local($line) {
  Say ""
  Say ("  本地完整跑 " + $line.name) Cyan
  Say "  本机可达全部行情源（腾讯/新浪/东财），比 GitHub runner 还多两个。" DarkGray
  Say "  注意：股票池和云端完全一样（cache\codes.csv，5548 只）。" DarkGray
  Say "  本地的优势是可用性，不是覆盖面。" DarkGray
  $envmap = Load-Env
  $canMail = ($envmap.ContainsKey("SMTP_HOST") -and $envmap.ContainsKey("MAIL_TO"))
  if ($canMail) { Say ("  已读到 tools\local.env，跑完发信到 " + $envmap["MAIL_TO"]) Green }
  else {
    Say "  没有 tools\local.env —— 跑完只出本地面板，不发邮件。" Yellow
    Say "  想本地也发信：把 tools\local.env.example 复制成 local.env 填好。" DarkGray
  }
  $a = Read-Host "  开始？(Y/n)"
  if ($a -eq "n") { return }
  $t0 = Get-Date

  if ($line.key -eq "auction") {
    $meta = Join-Path $ROOT "cache\universe_meta.json"
    $need = $true
    if (Test-Path $meta) {
      try {
        $j = Get-Content $meta -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($j.date -eq (BJNow).ToString("yyyy-MM-dd")) { $need = $false }
      } catch { }
    }
    if ($need) {
      Say "  [1/3] 候选池不是今天的，先建（3-5 分钟）..." Cyan
      Run-Py @("src\premarket.py") $envmap | Out-Null
    } else { Say "  [1/3] 候选池已是今天的，跳过" Gray }
    $bj = BJNow
    $lateArg = @()
    if ($bj.Hour -gt 9 -or ($bj.Hour -eq 9 -and $bj.Minute -ge 27)) {
      Say "  竞价窗口已过，用抢救模式（今开当竞价价）" Yellow
      $lateArg = @("--late")
    }
    Say "  [2/3] 采集 + 打分..." Cyan
    $c = Run-Py (@("src\run_auction.py","--stage","quick") + $lateArg) $envmap
    if ($c -ne 0) { Say "  采集失败，退出码 $c" Red; Pause; return }
    Say "  [3/3] 生成面板与附件..." Cyan
    Run-Py @("src\run_auction.py","--stage","enrich") $envmap | Out-Null
  } else {
    Say "  [1/2] 收盘后扫描..." Cyan
    $c = Run-Py @("src\pullback.py","--stage","scan") $envmap
    if ($c -ne 0) { Say "  扫描失败，退出码 $c" Red; Pause; return }
    Say "  [2/2] 生成面板与附件..." Cyan
    Run-Py @("src\pullback.py","--stage","send") $envmap | Out-Null
  }

  $sec = [int]((Get-Date) - $t0).TotalSeconds
  Say ("  完成，耗时 " + $sec + " 秒") Green
  $lm = Local-Meta $line
  if ($lm) { Say ("  本地产物：" + $lm.date + "，" + $lm.n + " 只") Green }
  $p = Join-Path $ROOT ($line.out + "\panel.html")
  if (Test-Path $p) {
    $a = Read-Host "  打开本地面板？(Y/n)"
    if ($a -ne "n") { Start-Process $p }
  }
  Pause
}

# --- 看日志 ------------------------------------------------------------------
function Show-Log {
  Say ""
  Say "  [1] 竞价云端日志   [2] 形态云端日志   [3] 本地最近一次运行输出" White
  $k = Read-Host "  选择"
  if ($k -eq "3") {
    $f = Join-Path $ROOT "tools\local_run.log"
    if (Test-Path $f) { Get-Content $f -Tail 40 -Encoding UTF8 | ForEach-Object { Say ("    " + $_) Gray } }
    else { Say "  还没有本地运行记录" Yellow }
    Pause; return
  }
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

# --- 打开面板 ----------------------------------------------------------------
function Open-Panel {
  Say ""
  Say "  [1] 在线竞价面板   [2] 在线形态面板   [3] 本地竞价   [4] 本地形态" White
  $k = Read-Host "  选择"
  if ($k -eq "1") { Start-Process $PAGE; return }
  if ($k -eq "2") { Start-Process ($PAGE + "pullback.html"); return }
  $dir = "out"; if ($k -eq "4") { $dir = "out_pullback" }
  if ($k -eq "3" -or $k -eq "4") {
    $p = Join-Path $ROOT ($dir + "\panel.html")
    if (Test-Path $p) { Start-Process $p }
    else { Say "  本地还没有这条线的产物，先用 [1]/[2] 里的「本地完整跑」" Yellow; Start-Sleep 2 }
  }
}

# --- 体检 --------------------------------------------------------------------
function Health {
  Clear-Host
  Say "  === 触发能力体检 ===" White; Say ""

  Say "  [电源策略]  决定机器睡着时任务能不能被叫醒" White
  $guid = ((powercfg /getactivescheme) -split " ")[3]
  $q = powercfg /query $guid SUB_SLEEP RTCWAKE
  $ac = (($q | Select-String "Current AC") -split ":")[1]
  $dc = (($q | Select-String "Current DC") -split ":")[1]
  $okWake = ($ac -match "0x00000001") -and ($dc -match "0x00000001")
  Write-Host "    唤醒定时器 插电/电池: " -NoNewline
  if ($okWake) { Say "启用 / 启用    正常" Green }
  else { Say "$ac / $dc    未全部启用，机器睡着时叫不醒" Red }

  Say ""
  Say "  [本机计划任务]" White
  foreach ($l in $LINES) {
    $t = Get-ScheduledTask -TaskName $l.task -ErrorAction SilentlyContinue
    Write-Host ("    " + $l.name + ": ") -NoNewline
    if (-not $t) { Say "不存在" Red; continue }
    $rep = $t.Triggers | Where-Object { $_.Repetition.Interval } | Select-Object -First 1
    $hasLogon = @($t.Triggers | Where-Object { $_.CimClass.CimClassName -like "*Logon*" }).Count -gt 0
    $b = @("状态 " + $t.State)
    if ($rep) { $b += ("重试 " + $rep.Repetition.Interval + "/" + $rep.Repetition.Duration) }
    else { $b += "无重试（睡着就错过）" }
    if ($hasLogon) { $b += "登录触发 有" } else { $b += "登录触发 无" }
    if (-not $t.Settings.DisallowStartIfOnBatteries) { $b += "电池可跑" } else { $b += "电池下不跑" }
    Say ($b -join " | ") Gray
  }

  Say ""
  Say "  [行情源可达性]  本地完整跑的前提" White
  $probe = @'
import urllib.request, socket, time
socket.setdefaulttimeout(10)
for name, url in [
  ("腾讯批量行情","https://qt.gtimg.cn/q=sh600000"),
  ("腾讯日K","https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600000,day,,,10,"),
  ("新浪交易日历","https://finance.sina.com.cn/realstock/company/sh000001/hisdata/klc_kl.js"),
  ("东财日线","https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600000&fields1=f1&fields2=f51&klt=101&fqt=0&end=20500101&lmt=5")]:
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"}))
        r.read()
        print("    OK   %-14s %.2fs" % (name, time.time()-t0))
    except Exception as e:
        print("    FAIL %-14s %s" % (name, type(e).__name__))
'@
  $env:PYTHONIOENCODING = "utf-8"
  $probe | & $PY - 2>&1 | ForEach-Object {
    if ("$_" -match "OK") { Say "$_" Green } else { Say "$_" Red } }

  Say ""
  Say "  [本地 Python 依赖]" White
  $dep = @'
import importlib
for m in ("pandas","pyarrow","yaml","requests","akshare"):
    try:
        importlib.import_module(m)
        print("    OK   " + m)
    except Exception:
        print("    MISS " + m + "    pip install " + m)
'@
  $dep | & $PY - 2>&1 | ForEach-Object {
    if ("$_" -match "OK") { Say "$_" Green } else { Say "$_" Red } }

  Say ""
  Say "  [本地发信配置]" White
  $e = Load-Env
  if ($e.ContainsKey("SMTP_HOST") -and $e.ContainsKey("MAIL_TO")) {
    Say ("    OK   tools\local.env 已配置，收件人 " + $e["MAIL_TO"]) Green }
  else {
    Say "    MISS tools\local.env 未配置 —— 本地跑只出面板，不发邮件" Yellow
    Say "         复制 tools\local.env.example 成 local.env 填好即可" DarkGray }

  Say ""
  Say "  唤醒定时器只能叫醒**睡眠**中的机器。完全关机时本地方案一律无效，" DarkGray
  Say "  那种情况靠 GitHub 自接力（6-接力守护）和 Cloudflare 外部触发。" DarkGray
  Pause
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
    "5" { Open-Panel }
    "6" { Health }
    "0" { exit 0 }
    default { }
  }
}
