#!/bin/bash
declare -a tasks_0=(
	"python train_item.py  --dataset ours --epochs=10 --idx 70  --tokens 10  --omega 1.0  --topk 256  --space 0"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 71  --tokens 6  --omega 1.0  --topk 256  --space 0"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 72  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 73  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 74  --tokens 6  --omega 1.0  --topk 256  --space 0"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 75  --tokens 6  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 76  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 76  --tokens 10  --omega 1.0  --topk 64  --space 0"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 77  --tokens 10  --omega 0.1  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 78  --tokens 6  --omega 1.0  --topk 64  --space 1"\ 

)
declare -a tasks_1=(
	"python train_item.py  --dataset ours --epochs=10 --idx 78  --tokens 8  --omega 1.0  --topk 64  --space 0"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 79  --tokens 6  --omega 1.0  --topk 64  --space 0"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 80  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 80  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 81  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 81  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 82  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 83  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 84  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 85  --tokens 10  --omega 1.0  --topk 256  --space 0"\ 

)
declare -a tasks_2=(
	"python train_item.py  --dataset ours --epochs=10 --idx 86  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 87  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 87  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 88  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 88  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 89  --tokens 5  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 90  --tokens 6  --omega 1.0  --topk 32  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 91  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 92  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 93  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 

)
declare -a tasks_3=(
	"python train_item.py  --dataset ours --epochs=10 --idx 93  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 94  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 95  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 95  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 96  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 97  --tokens 6  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 98  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 99  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 100  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 101  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 

)
declare -a tasks_4=(
	"python train_item.py  --dataset ours --epochs=10 --idx 101  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 102  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 103  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 104  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 105  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 106  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 107  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 108  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 109  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 110  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 

)
declare -a tasks_5=(
	"python train_item.py  --dataset ours --epochs=10 --idx 111  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 112  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 113  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 114  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 115  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 116  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 117  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 117  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 118  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 119  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 

)
declare -a tasks_6=(
	"python train_item.py  --dataset ours --epochs=10 --idx 120  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 121  --tokens 1  --omega 1.0  --topk 0  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 121  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 122  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 122  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 123  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 124  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 125  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 126  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 127  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 

)
declare -a tasks_7=(
	"python train_item.py  --dataset ours --epochs=10 --idx 128  --tokens 8  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 128  --tokens 6  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 129  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 130  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 130  --tokens 6  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 131  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 132  --tokens 8  --omega 1.0  --topk 256  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 133  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 133  --tokens 10  --omega 1.0  --topk 64  --space 1"\ 
	"python train_item.py  --dataset ours --epochs=10 --idx 134  --tokens 10  --omega 1.0  --topk 256  --space 1"\ 

)
run_tasks() {
    local device=$1
    local tasks=("${@:2}")
    
    for task in "${tasks[@]}"; do
        CUDA_VISIBLE_DEVICES=$device $task
    done
}

run_tasks 0 "${tasks_0[@]}" &
run_tasks 1 "${tasks_1[@]}" &
run_tasks 2 "${tasks_2[@]}" &
run_tasks 3 "${tasks_3[@]}" &
run_tasks 4 "${tasks_4[@]}" &
run_tasks 5 "${tasks_5[@]}" &
run_tasks 6 "${tasks_6[@]}" &
run_tasks 7 "${tasks_7[@]}" &

wait

echo "All tasks have completed."
