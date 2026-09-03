' 无头启动器：把任意命令跑在完全不可见的窗口里。
'
' 计划任务直接指到 cmd.exe 会闪一个黑框（哪怕任务设了隐藏，cmd 自己
' 还是会创建控制台窗口）。wscript 跑 .vbs 没有控制台，由它再去启动
' 目标命令并指定 0 号窗口样式，就彻底看不见了。
'
' 用法:  wscript.exe //B headless.vbs "C:\...\trigger_auction.cmd"
' 第二个参数起原样传给目标命令。
'
' 注意 Run 的第三个参数是 True：等目标跑完再退出，这样计划任务的
' 「上次运行结果」才反映真实退出码，而不是永远 0。
Option Explicit
Dim sh, cmd, i
If WScript.Arguments.Count < 1 Then WScript.Quit 2
cmd = """" & WScript.Arguments(0) & """"
For i = 1 To WScript.Arguments.Count - 1
  cmd = cmd & " """ & WScript.Arguments(i) & """"
Next
Set sh = CreateObject("WScript.Shell")
WScript.Quit sh.Run(cmd, 0, True)
