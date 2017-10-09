import torch
from torch.utils import data
from scipy.io import loadmat

class Tree(object):
    class Node(object):
        def __init__(self, leaf=None, left=None, right=None, nType=None, sym=None):
            self.box = leaf
            self.sym = sym
            self.left = left
            self.right = right
            self.nType = nType
            if nType == 0:
                self.label = torch.FloatTensor([1,0,0]).unsqueeze(0)
            if nType == 1:
                self.label = torch.FloatTensor([0,1,0]).unsqueeze(0)
            if nType == 2:
                self.label = torch.FloatTensor([0,0,1]).unsqueeze(0)

        def is_leaf(self):
            return self.box is not None

    def __init__(self, boxes, ops, syms):
        buffer = [b for b in torch.split(boxes, 1, 0)]
        sympara = [s for s in torch.split(syms, 1, 0)]
        opnum = ops.size()[1]
        queue = []
        buffer.reverse()
        sympara.reverse()
        for i in range(opnum):
            if ops[0, i] == 0:
                queue.append(Tree.Node(leaf=buffer.pop(), nType=0))
            if ops[0, i] == 1:
                n_left = queue.pop()
                n_right = queue.pop()
                queue.append(Tree.Node(left=n_left, right=n_right, nType=1))
            if ops[0, i] == 2:
                n_left = queue.pop()
                queue.append(Tree.Node(left=n_left, sym=sympara.pop(), nType=2))
        assert len(queue) == 1
        self.root = queue[0]


class GRASSDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        box_data = torch.from_numpy(loadmat(self.dir+'/box_data.mat')['boxes']).float()
        op_data = torch.from_numpy(loadmat(self.dir+'/op_data.mat')['ops']).int()
        sym_data = torch.from_numpy(loadmat(self.dir+'/sym_data.mat')['syms']).float()
        #weight_list = torch.from_numpy(loadmat(self.dir+'/weights.mat')['weights']).float()
        num_examples = op_data.size()[1]
        box_data = torch.chunk(box_data, num_examples, 1)
        op_data = torch.chunk(op_data, num_examples, 1)
        sym_data = torch.chunk(sym_data, num_examples, 1)
        #weight_list = torch.chunk(weight_list, num_examples, 1)
        self.transform = transform
        self.trees = []
        for i in range(len(op_data)) :
            boxes = torch.t(box_data[i])
            ops = torch.t(op_data[i])
            syms = torch.t(sym_data[i])
            tree = Tree(boxes, ops, syms)
            self.trees.append(tree)

    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)