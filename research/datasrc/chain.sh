#!/bin/bash
# hotrank 跑完再跑 hotprofile（同一路东财配额，不并行）
cd /c/home/daily-report/research/datasrc
until grep -q "hotrank 结束" rescue_hotrank.log 2>/dev/null; do sleep 30; done
python rescue_perstock.py hotprofile > hotprofile.out 2>&1
