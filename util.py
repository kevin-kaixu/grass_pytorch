import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='grass_pytorch')
    parser.add_argument('--box_code_size', type=int, default=12)
    parser.add_argument('--feature_size', type=int, default=80)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--symmetry_size', type=int, default=8)
    parser.add_argument('--max_box_num', type=int, default=30)
    parser.add_argument('--max_sym_num', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--save_snapshot', action='store_false', default=True)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_by', type=float, default=1)
    parser.add_argument('--lr_decay_every', type=float, default=1)

    parser.add_argument('--cuda', action='store_false', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args