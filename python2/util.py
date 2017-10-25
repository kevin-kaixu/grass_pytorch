from __future__ import absolute_import
import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description=u'grass_pytorch')
    parser.add_argument(u'--box_code_size', type=int, default=12)
    parser.add_argument(u'--feature_size', type=int, default=80)
    parser.add_argument(u'--hidden_size', type=int, default=200)
    parser.add_argument(u'--symmetry_size', type=int, default=8)
    parser.add_argument(u'--max_box_num', type=int, default=30)
    parser.add_argument(u'--max_sym_num', type=int, default=10)

    parser.add_argument(u'--epochs', type=int, default=300)
    parser.add_argument(u'--batch_size', type=int, default=123)
    parser.add_argument(u'--show_log_every', type=int, default=3)
    parser.add_argument(u'--save_log', action=u'store_true', default=False)
    parser.add_argument(u'--save_log_every', type=int, default=3)
    parser.add_argument(u'--save_snapshot', action=u'store_true', default=False)
    parser.add_argument(u'--save_snapshot_every', type=int, default=5)
    parser.add_argument(u'--no_plot', action=u'store_true', default=False)
    parser.add_argument(u'--lr', type=float, default=.001)
    parser.add_argument(u'--lr_decay_by', type=float, default=1)
    parser.add_argument(u'--lr_decay_every', type=float, default=1)

    parser.add_argument(u'--no_cuda', action=u'store_true', default=False)
    parser.add_argument(u'--gpu', type=int, default=0)
    parser.add_argument(u'--data_path', type=unicode, default=u'data')
    parser.add_argument(u'--save_path', type=unicode, default=u'models')
    parser.add_argument(u'--resume_snapshot', type=unicode, default=u'')
    args = parser.parse_args()
    return args