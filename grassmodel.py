import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time

#########################################################################################
## Encoder
#########################################################################################

class Sampler(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(featureSize, hiddenSize)
        self.mlp2mu = nn.Linear(hiddenSize, featureSize)
        self.mlp2log = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()
        
    def forward(self,input):
        encode = self.tanh(self.mlp1(input))
        mu = self.mlp2mu(encode)
        logvar = self.mlp2log(encode)
        
        std = logvar.mul(0.5).exp_() #calculate the STDEV
        eps = torch.FloatTensor(std.size()).normal_().cuda() #random normalized noise
        eps = Variable(eps)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.cat([eps.mul(std).add_(mu), KLD_element], 1)

class BoxEncoder(nn.Module):

    def __init__(self, boxSize, featureSize):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(boxSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, boxes_in):
        boxes = self.encoder(boxes_in)
        boxes = self.tanh(boxes)
        return boxes

class AdjEncoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(featureSize, hiddenSize, bias=False)
        self.second = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        out = self.left(left_in)
        out += self.right(right_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        return out

class SymEncoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(featureSize, hiddenSize)
        self.right = nn.Linear(symmetrySize, hiddenSize)
        self.second = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, left_in, right_in):
        out = self.left(left_in)
        out += self.right(right_in)
        out = self.tanh(out)
        out = self.second(out)
        out = self.tanh(out)
        return out

class GRASSEncoder(nn.Module):

    def __init__(self, config):
        super(GRASSEncoder, self).__init__()
        self.boxEncoder = BoxEncoder(boxSize = config.boxSize, featureSize = config.featureSize)
        self.adjEncoder = AdjEncoder(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.symEncoder = SymEncoder(featureSize = config.featureSize, symmetrySize = config.symmetrySize, hiddenSize = config.hiddenSize)
        self.sampler = Sampler(featureSize = config.featureSize, hiddenSize = config.hiddenSize)

    def leafNode(self, box):
        return self.boxEncoder(box)

    def adjNode(self, left, right):
        return self.adjEncoder(left, right)

    def symNode(self, feature, sym):
        return self.symEncoder(feature, sym)

    def sampleLayer(self, feature):
        return self.sampler(feature)

def encode_structure_fold(fold, tree):

    def encode_node(node):
        if node.is_leaf():
            return fold.add('leafNode', node.box)
        if node.nType == 1:
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjNode', left, right)
        if node.nType == 2:
            feature = encode_node(node.left)
            sym = node.sym
            return fold.add('symNode', feature, sym)

    encoding = encode_node(tree.root)
    return fold.add('sampleLayer', encoding)

#########################################################################################
## Decoder
#########################################################################################

class NodeClassifier(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(NodeClassifier, self).__init__()
        self.first = nn.Linear(featureSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.second = nn.Linear(hiddenSize, 3)
        self.softmax = nn.Softmax()

    def forward(self, feature):
        out = self.first(feature)
        out = self.tanh(out)
        out = self.second(out)
        out = self.softmax(out)
        return out

class Desampler(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(Desampler, self).__init__()
        self.mlp1 = nn.Linear(featureSize, hiddenSize)
        self.mlp2 = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()
        
    def forward(self,input):
        output = self.tanh(self.mlp1(input))
        output = self.tanh(self.mlp2(output))
        return output

class AdjDecoder(nn.Module):

    def __init__(self, featureSize, hiddenSize):
        super(AdjDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, featureSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        l = self.left(out)
        r = self.right(out)
        l = self.tanh(l)
        r = self.tanh(r)
        return l, r

class SymDecoder(nn.Module):

    def __init__(self, featureSize, symmetrySize, hiddenSize):
        super(SymDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, hiddenSize)
        self.tanh = nn.Tanh()
        self.left = nn.Linear(hiddenSize, featureSize)
        self.right = nn.Linear(hiddenSize, symmetrySize)

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        f = self.left(out)
        f = self.tanh(f)
        s = self.right(out)
        s = self.tanh(s)
        return f, s

class BoxDecoder(nn.Module):

    def __init__(self, boxSize, featureSize):
        super(BoxDecoder, self).__init__()
        self.decode = nn.Linear(featureSize, boxSize)
        self.tanh = nn.Tanh()

    def forward(self, parent_in):
        out = self.decode(parent_in)
        out = self.tanh(out)
        return out

class GRASSDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSDecoder, self).__init__()
        self.boxDecoder = BoxDecoder(boxSize = config.boxSize, featureSize = config.featureSize)
        self.adjDecoder = AdjDecoder(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.symDecoder = SymDecoder(featureSize = config.featureSize, symmetrySize = config.symmetrySize, hiddenSize = config.hiddenSize)
        self.desampler = Desampler(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.nodeClassifier = NodeClassifier(featureSize = config.featureSize, hiddenSize = config.hiddenSize)
        self.mseLoss = nn.MSELoss()

    def leafNode(self, feature):
        return self.boxDecoder(feature)

    def adjNode(self, feature):
        return self.adjDecoder(feature)

    def symNode(self, feature):
        return self.symDecoder(feature)

    def deSampleLayer(self, feature):
        return self.desampler(feature)

    def mseBoxLossLayer(self, f1, f2):
        f2 = Variable(f2).cuda()
        loss = ((f1.add(-f2))**2).sum(1).mul(0.4)
        return loss

    def mseSymLossLayer(self, f1, f2):
        f2 = Variable(f2).cuda()
        loss = ((f1.add(-f2))**2).sum(1).mul(0.5)
        return loss

    def addLayer(self, f1, f2):
        return f1.add_(f2)

    def classLossLayer(self, f1, f2):
        f = self.nodeClassifier(f1)
        f2 = Variable(f2).cuda()
        return torch.log(f).mul(f2).sum(1).mul(-0.2)

    def classLayer(self, f):
        l = self.nodeClassifier(f)
        _, op = torch.max(l, 1)
        return op

def decode_structure_fold(fold, feature, tree):

    def decode_node_box(node, feature):
        if node.nType == 0:
            box = fold.add('leafNode', feature)
            loss = fold.add('mseBoxLossLayer', box, node.box)
            label = fold.add('classLossLayer', feature, node.label)
            return fold.add('addLayer', loss, label)

        if node.nType == 1:
            left, right = fold.add('adjNode', feature).split(2)
            l = decode_node_box(node.left, left)
            r = decode_node_box(node.right, right)
            label = fold.add('classLossLayer', feature, node.label)
            l = fold.add('addLayer', l, label)
            return fold.add('addLayer', l, r)

        if node.nType == 2:
            f, sym = fold.add('symNode', feature).split(2)
            loss = fold.add('mseSymLossLayer', sym, node.sym)
            l = decode_node_box(node.left, f)
            label = fold.add('classLossLayer', feature, node.label)
            l = fold.add('addLayer', l, label)
            return fold.add('addLayer', l, loss)

    decode = fold.add('deSampleLayer', feature)
    loss = decode_node_box(tree.root, decode)
    return loss

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    return m

def decode_structure(model, feature):
    decode = model.deSampleLayer(feature)
    syms = [torch.ones(8).mul(10).cuda()]
    stack = [decode]
    boxes = []
    while len(stack) > 0:
        f = stack.pop()
        label = model.classLayer(f)
        label = label.data
        if label[0] == 1:
            left, right = model.adjNode(f)
            stack.append(left)
            stack.append(right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if label[0] == 2:
            left, s = model.symNode(f)
            s = s.squeeze(0)
            stack.append(left)
            syms.pop()
            syms.append(s.data)
        if label[0] == 0:
            reBox = model.leafNode(f)
            reBoxes = [reBox]
            s = syms.pop()
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            if l1 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7])
                for i in range(folds-1):
                    rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    reBoxes.append(Variable(newbox.unsqueeze(0)))

            if l2 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans**2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
                folds = round(trans_total/trans_length)
                for i in range(folds):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i+1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    reBoxes.append(Variable(newbox.unsqueeze(0)))

            if l3 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2*ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2*ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                reBoxes.append(Variable(newbox.unsqueeze(0)))

            boxes.extend(reBoxes)

    return boxes
            

        