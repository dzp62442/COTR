#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}  # 定义了参与训练的节点数，默认为1
NODE_RANK=${NODE_RANK:-0}  # 当前节点的标记，从0开始计数，默认为0。在单节点训练时，使用默认值
PORT=${PORT:-29500}  # 主节点监听的数据通信端口，默认为29500
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  # 设置主节点的IP地址，单节点训练默认使用本地回环地址（即127.0.0.1）

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_occ.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}  # 将所有从第3个开始的脚本参数传递给主训练脚本