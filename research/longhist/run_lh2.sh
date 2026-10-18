#!/bin/sh
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
python research/blind/walk_lh.py --mode fixed > research/blind/walk_lh_fixed.log 2>&1
python research/blind/walk_lh.py --mode roll36 > research/blind/walk_lh_roll36.log 2>&1
python research/blind/walk_lh.py --mode rollall > research/blind/walk_lh_rollall.log 2>&1
touch research/longhist/lh_done.flag
