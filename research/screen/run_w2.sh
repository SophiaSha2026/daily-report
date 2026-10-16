#!/bin/bash
# 等到北京 18:31 再开跑（16:00~18:30 生产在算特征）
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
while true; do
  hm=$(python -c "import sys;sys.path.insert(0,'src');from bjtime import now_bj;t=now_bj();print(t.hour*60+t.minute)")
  if [ "$hm" -ge 1111 ] || [ "$hm" -lt 960 ]; then break; fi
  sleep 300
done
echo "开跑 $(python -c "import sys;sys.path.insert(0,'src');from bjtime import now_bj;print(now_bj())")"
python research/screen/exp_screen.py --fit --name screen_w2_v1 --arms base eqd evw y_rel4 y_rel2 y_softh y_softz y_rel4_eqd --seeds 7
echo "退出码 $?"
