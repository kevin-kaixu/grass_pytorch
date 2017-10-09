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

class GRASS(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        o = torch.from_numpy(loadmat(root+'/ops.mat')['ops']).int()
        b = torch.from_numpy(loadmat(root+'/boxes.mat')['boxes']).float()
        s = torch.from_numpy(loadmat(root+'/syms.mat')['syms']).float()
        w = torch.from_numpy(loadmat(root+'/weights.mat')['weights']).float()
        l = o.size()[1]
        self.opData = torch.chunk(o,l,1)
        self.boxData = torch.chunk(b,l,1)
        self.symData = torch.chunk(s,l,1)
        self.wData = torch.chunk(w,l,1)
        self.transform = transform
        self.trees = []
        for i in range(len(self.opData)):
            box = torch.t(self.boxData[i])
            op = torch.t(self.opData[i])
            sym = torch.t(self.symData[i])
            tree = Tree(box, op, sym)
            self.trees.append(tree)

    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.opData)