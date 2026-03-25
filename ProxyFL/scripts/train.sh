#!/bin/bash

# 默认参数值
PYTHON_SCRIPT="proxyfl.py"
DATASET="CIFAR100"
ALPHA=0.1
GPU_ID=7
METHOD="proxyfl"
LR_SERVER=0.005
BS_SERVER=4
SERVER_EPOCHS=100
TOTAL_SERVER_EPOCHS=30000
LOCAL_EPOCHS=5
NUM_ONLINE_CLIENTS=8
NUM_CLIENTS=20
TOPK=0.2
THRES=0.95

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s| --script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        -d| --dataset)
            DATASET="$2"
            shift 2
            ;;
        -a| --alpha)
            ALPHA="$2"
            shift 2
            ;;
        -g| --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -lrs| --lr_server)
            LR_SERVER="$2"
            shift 2
            ;;
        -bss| --bs_server)
            BS_SERVER="$2"
            shift 2
            ;;
        -ep_s| --server_epochs)
            SERVER_EPOCHS="$2"
            shift 2
            ;;
        -ep_all| --total_server_epochs)
            TOTAL_SERVER_EPOCHS="$2"
            shift 2
            ;;
        -tao| --threshold)
            THRES="$2"
            shift 2
            ;;
        -ep_loc| --local_epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        -n_on| --num_online_clients)
            NUM_ONLINE_CLIENTS="$2"
            shift 2
            ;;
        -n| --num_clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        -m| --method)
            METHOD="$2"
            shift 2
            ;;
        -t| --topk)
            TOPK="$2"
            shift 2
            ;;    
        -h| --help)
            show_help
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done


# 执行 Python 命令
python "$PYTHON_SCRIPT" \
    --dataset "$DATASET" \
    --alpha "$ALPHA" \
    --gpu_id "$GPU_ID" \
    --method "$METHOD" \
    --lr_server "$LR_SERVER" \
    --bs_server "$BS_SERVER" \
    --server_epochs "$SERVER_EPOCHS" \
    --total_server_epochs "$TOTAL_SERVER_EPOCHS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --num_online_clients "$NUM_ONLINE_CLIENTS" \
    --num_clients "$NUM_CLIENTS" \
    --topk "$TOPK" \
    --threshold "$THRES" \


# bash scripts/train.sh --dataset CIFAR100 --alpha 0.1 --gpu 0
# bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu 1
# bash scripts/train.sh --dataset SVHN --alpha 0.1 --gpu 2
# bash scripts/train.sh --dataset CINIC10 --alpha 0.1 --gpu 3