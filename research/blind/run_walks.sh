#!/bin/sh
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
python research/blind/walk_bl.py --name v2_lag20 --mode lag > research/blind/walk_v2_lag20.log 2>&1
python research/blind/walk_bl.py --name fe_nomkt --mode fe --feats price_nomkt > research/blind/walk_fe_nomkt.log 2>&1
python research/blind/walk_bl.py --name reg_l2 --mode reg > research/blind/walk_reg_l2.log 2>&1
python research/blind/walk_bl.py --name q90 --mode q90 > research/blind/walk_q90.log 2>&1
python research/blind/walk_bl.py --name multi3 --mode multi > research/blind/walk_multi3.log 2>&1
python research/blind/walk_bl.py --name fe_nomkt_shufd --mode fe --feats price_nomkt --shuffle day > research/blind/walk_fe_nomkt_shufd.log 2>&1
touch research/blind/walks_done.flag
