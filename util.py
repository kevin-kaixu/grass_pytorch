import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='PyTorch/GRASS')
    parser.add_argument('--boxSize', type=int, default=12)
    parser.add_argument('--featureSize', type=int, default=80)
    parser.add_argument('--hiddenSize', type=int, default=200)
    parser.add_argument('--symmetrySize', type=int, default=8)
    parser.add_argument('--maxBoxes', type=int, default=30)
    parser.add_argument('--maxSyms', type=int, default=10)
    args = parser.parse_args()
    return args