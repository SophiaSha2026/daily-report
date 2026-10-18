#!/bin/bash
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
python research/gate/walk_gt.py --name lgb_price_top50 --model lgb --feats price --universe top50 --neg 1.0 > research/gate/walk_lgb_price_top50.log 2>&1
python research/gate/walk_gt.py --name lgb_alt_top50 --model lgb --feats alt --universe top50 --neg 1.0 > research/gate/walk_lgb_alt_top50.log 2>&1
python research/gate/walk_gt.py --name lr_core_top50 --model lr --feats lr --universe top50 --neg 1.0 > research/gate/walk_lr_core_top50.log 2>&1
python research/gate/walk_gt.py --name lgb_core_top50 --model lgb --feats core --universe top50 --neg 1.0 > research/gate/walk_lgb_core_top50.log 2>&1
echo DONE3 > research/gate/walks3_done.flag
