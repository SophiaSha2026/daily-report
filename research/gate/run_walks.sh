#!/bin/bash
# 顺序跑各配置的滚动（内存有限，不并行）
cd /c/home/daily-report
export PYTHONIOENCODING=utf-8
while pgrep -f "walk_gt.py --name lgb_price " >/dev/null; do sleep 20; done
python research/gate/walk_gt.py --name lgb_alt --model lgb --feats alt > research/gate/walk_lgb_alt.log 2>&1
python research/gate/walk_gt.py --name lr_core --model lr --feats lr > research/gate/walk_lr_core.log 2>&1
python research/gate/walk_gt.py --name lgb_price_deep --model lgb --feats price --depth deep > research/gate/walk_lgb_price_deep.log 2>&1
python research/gate/walk_gt.py --name lgb_price_top300 --model lgb --feats price --universe top300 --neg 1.0 > research/gate/walk_lgb_price_top300.log 2>&1
python research/gate/walk_gt.py --name lgb_core --model lgb --feats core > research/gate/walk_lgb_core.log 2>&1
python research/gate/walk_gt.py --name lgb_price_nomkt --model lgb --feats price_nomkt > research/gate/walk_lgb_price_nomkt.log 2>&1
python research/gate/walk_gt.py --name lgb_price_shufg --model lgb --feats price --shuffle global > research/gate/walk_shufg.log 2>&1
python research/gate/walk_gt.py --name lgb_price_shufd --model lgb --feats price --shuffle day > research/gate/walk_shufd.log 2>&1
echo ALLDONE > research/gate/walks_done.flag
