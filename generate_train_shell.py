import os
import json
from config.param import PARAMS

with open('./shell/train.sh', 'w') as f:
    f.write('#!/bin/bash')
    f.write('\n')

target_codes = json.load(open('./ref/target_list.json', 'r'))
train_shell = []
for idx in range(0, 70):
    target_code = target_codes[idx]
    param_list = PARAMS[target_code]
    for param in param_list:
        train_idx = idx + 70
        tokens, topk, omega, use_space = int(param[0]), int(param[1]), float(param[2]), str(param[3])
        train_ = f'python train_item.py  --dataset ours  --idx {train_idx}  --tokens {tokens}  --omega {omega}  --topk {topk}  --space {1 if use_space=="T" else 0}'
        train_shell.append(train_)

train_exp_nums = len(train_shell)
step = len(train_shell) // 8

for gpu_id in range(0, 8):
    train_shell_gpu = train_shell[step*gpu_id:step*(gpu_id+1)]
    task = ''
    for p in train_shell_gpu:
        task += f'\t\"{p}\"\\ \n'
    d = f'declare -a tasks_{gpu_id}=(\n{task}\n)'

    with open('./shell/train.sh', 'a') as f:
        f.write(d)
        f.write('\n')

runs = [
"run_tasks() {\n",
"    local device=$1\n",
"    local tasks=(\"${@:2}\")\n",
"    \n",
"    for task in \"${tasks[@]}\"; do\n",
"        CUDA_VISIBLE_DEVICES=$device $task\n",
"    done\n",
"}\n",
"\n",
"run_tasks 0 \"${tasks_0[@]}\" &\n",
"run_tasks 1 \"${tasks_1[@]}\" &\n",
"run_tasks 2 \"${tasks_2[@]}\" &\n",
"run_tasks 3 \"${tasks_3[@]}\" &\n",
"run_tasks 4 \"${tasks_4[@]}\" &\n",
"run_tasks 5 \"${tasks_5[@]}\" &\n",
"run_tasks 6 \"${tasks_6[@]}\" &\n",
"run_tasks 7 \"${tasks_7[@]}\" &\n",
"\n",
"wait\n",
"\n",
"echo \"All tasks have completed.\"\n",
]

with open('./shell/train.sh', 'a') as f:
    for run in runs:
        f.write(run)