#!/bin/sh
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
until grep -q "完成" research/longhist/fetch.log; do sleep 30; done
python research/longhist/build_lh.py --stage feat > research/longhist/build_feat2.log 2>&1
python research/longhist/build_lh.py --stage day > research/longhist/build_day.log 2>&1
python research/longhist/build_lh.py --stage table > research/longhist/build_table.log 2>&1
python research/longhist/build_lh.py --stage check > research/longhist/build_check.log 2>&1
python research/blind/walk_lh.py --mode fixed > research/blind/walk_lh_fixed.log 2>&1
python research/blind/walk_lh.py --mode roll36 > research/blind/walk_lh_roll36.log 2>&1
python research/blind/walk_lh.py --mode rollall > research/blind/walk_lh_rollall.log 2>&1
touch research/longhist/lh_done.flag
