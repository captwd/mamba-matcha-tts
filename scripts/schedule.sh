#!/bin/bash
# 【中文说明】批量排程脚本：依次执行多个训练任务（不同 epochs 的消融实验）
# 在项目根目录运行：bash scripts/schedule.sh
# 注意：此脚本为模板（原样引用了 src/train.py），实际使用时需把 src/train.py 改成 matcha/train.py

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=5 logger=csv

python src/train.py trainer.max_epochs=10 logger=csv
