#!/bin/bash
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
while [ ! -f research/gate/scores/lgb_alt.npy ]; do sleep 20; done
python research/gate/walk_gt.py --name lgb_price_top300 --model lgb --feats price --universe top300 --neg 1.0 > research/gate/walk_lgb_price_top300.log 2>&1
python research/gate/walk_gt.py --name lgb_alt_top300 --model lgb --feats alt --universe top300 --neg 1.0 > research/gate/walk_lgb_alt_top300.log 2>&1
python research/gate/walk_gt.py --name lgb_nomkt_top300 --model lgb --feats price_nomkt --universe top300 --neg 1.0 > research/gate/walk_lgb_nomkt_top300.log 2>&1
python research/gate/walk_gt.py --name lr_core_top300 --model lr --feats lr --universe top300 --neg 1.0 > research/gate/walk_lr_core_top300.log 2>&1
python research/gate/walk_gt.py --name lr_core --model lr --feats lr > research/gate/walk_lr_core.log 2>&1
python research/gate/walk_gt.py --name lgb_price_nomkt --model lgb --feats price_nomkt > research/gate/walk_lgb_price_nomkt.log 2>&1
python research/gate/walk_gt.py --name lgb_top300_shufg --model lgb --feats price --universe top300 --neg 1.0 --shuffle global > research/gate/walk_shufg.log 2>&1
python research/gate/walk_gt.py --name lgb_top300_shufd --model lgb --feats price --universe top300 --neg 1.0 --shuffle day > research/gate/walk_shufd.log 2>&1
python research/gate/walk_gt.py --name lgb_price_deep_top300 --model lgb --feats price --universe top300 --neg 1.0 --depth deep > research/gate/walk_lgb_price_deep_top300.log 2>&1
echo ALLDONE > research/gate/walks_done.flag
