#!/bin/bash

#block(name=gen_pose,threads=5, memory=60000, subtasks=1, gpu=true, hours=666)

python -u imgs2poses.py --scenedir /home/jinzhi/hdd10T/space4New/zhangkai/TMP_SRC/Train --match_type exhaustive_matcher
