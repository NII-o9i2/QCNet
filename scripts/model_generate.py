import torch
import pytorch_lightning as pl

import sys
import os

# 获取父目录的路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将父目录添加到sys.path
sys.path.append(parent_dir)

from datamodules import ArgoverseV2DataModule
from predictors import QCNet
from argparse import ArgumentParser
from torch_geometric.data import HeteroData

def create_dummy_hetero_data():
    data = HeteroData()
    # data['x1'].x = torch.randn(10, 32)  # 10个节点，每个节点32维特征
    # data['x2'].x = torch.randn(20, 64)  # 20个节点，每个节点64维特征
    # data['x1', 'edge1', 'x2'].edge_index = torch.randint(0, 10, (2, 30))  # 边连接x1和x2
    # data['x2', 'edge2', 'x1'].edge_index = torch.randint(0, 20, (2, 40))  # 边连接x2和x1

    return data


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))

    # 加载检查点
    checkpoint_path = "/home/SENSETIME/fengxiaotong/Desktop/code/autopilot-model/qcnet/pre_train_raw/QCNet_AV2.ckpt"
    checkpoint = torch.load(checkpoint_path)
    #
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    #
    # # 保存为.pt文件
    output_path = "/home/SENSETIME/fengxiaotong/Desktop/code/autopilot-model/qcnet/pre_train_raw/QCNet_AV2.pt"
    torch.save(model.state_dict(), output_path)
    #
    print(f"模型已保存为 {output_path}")
