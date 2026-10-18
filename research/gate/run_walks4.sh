#!/bin/bash
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
while [ ! -f research/gate/walks_done.flag ]; do sleep 20; done
python research/gate/walk_gt.py --name lgb_top300_shufg --model lgb --feats price --universe top300 --neg 1.0 --shuffle global > research/gate/walk_shufg.log 2>&1
python research/gate/walk_gt.py --name lgb_price_top20 --model lgb --feats price --universe top20 --neg 1.0 > research/gate/walk_lgb_price_top20.log 2>&1
echo DONE4 > research/gate/walks4_done.flag
