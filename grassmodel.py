import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time

#########################################################################################
## Encoder
#########################################################################################

class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        encode = self.tanh(self.mlp1(input))
        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)
        std = logvar.mul(0.5).exp_() # calculate the STDEV
        eps = Variable(torch.FloatTensor(std.size()).normal_().cuda()) # random normalized noise
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        return torch.cat([eps.mul(std).add_(mu), KLD_element], 1)

class BoxEncoder(nn.Module):

    def __init__(self, input_size, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, box_input):
        box_vector = self.encoder(box_input)
        box_vector = self.tanh(box_vector)
        return box_vector

class AdjEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(AdjEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(feature_size, hidden_size, bias=False)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output

class SymEncoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(symmetry_size, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output

class GRASSEncoder(nn.Module):

    def __init__(self, config):
        super(GRASSEncoder, self).__init__()
        self.box_encoder = BoxEncoder(input_size = config.box_code_size, feature_size = config.feature_size)
        self.adj_encoder = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.sym_encoder = SymEncoder(feature_size = config.feature_size, symmetry_size = config.symmetry_size, hidden_size = config.hidden_size)
        self.sample_encoder = Sampler(feature_size = config.feature_size, hidden_size = config.hidden_size)

    def boxEncoder(self, box):
        return self.box_encoder(box)

    def adjEncoder(self, left, right):
        return self.adj_encoder(left, right)

    def symEncoder(self, feature, sym):
        return self.sym_encoder(feature, sym)

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

def encode_structure_fold(fold, tree):

    def encode_node(node):
        if node.is_leaf():
            return fold.add('boxEncoder', node.box)
        elif node.is_adj():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder', left, right)
        elif node.is_sym():
            feature = encode_node(node.left)
            sym = node.sym
            return fold.add('symEncoder', feature, sym)

    encoding = encode_node(tree.root)
    return fold.add('sampleEncoder', encoding)

#########################################################################################
## Decoder
#########################################################################################

class NodeClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(hidden_size, 3)
        #self.softmax = nn.Softmax()

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        #output = self.softmax(output)
        return output

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input_feature):
        output = self.tanh(self.mlp1(input_feature))
        output = self.tanh(self.mlp2(output))
        return output

class AdjDecoder(nn.Module):
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, feature_size, hidden_size):
        super(AdjDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.mlp_left = nn.Linear(hidden_size, feature_size)
        self.mlp_right = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        left_feature = self.mlp_left(vector)
        left_feature = self.tanh(left_feature)
        right_feature = self.mlp_right(vector)
        right_feature = self.tanh(right_feature)
        return left_feature, right_feature

class SymDecoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size) # layer for decoding a feature vector 
        self.tanh = nn.Tanh()
        self.mlp_sg = nn.Linear(hidden_size, feature_size) # layer for outputing the feature of symmetry generator
        self.mlp_sp = nn.Linear(hidden_size, symmetry_size) # layer for outputing the vector of symmetry parameter

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        sym_gen_vector = self.mlp_sg(vector)
        sym_gen_vector = self.tanh(sym_gen_vector)
        sym_param_vector = self.mlp_sp(vector)
        sym_param_vector = self.tanh(sym_param_vector)
        return sym_gen_vector, sym_param_vector

class BoxDecoder(nn.Module):

    def __init__(self, feature_size, box_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, box_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        return vector

class GRASSDecoder(nn.Module):
    def __init__(self, config):
        super(GRASSDecoder, self).__init__()
        self.box_decoder = BoxDecoder(feature_size = config.feature_size, box_size = config.box_code_size)
        self.adj_decoder = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.sym_decoder = SymDecoder(feature_size = config.feature_size, symmetry_size = config.symmetry_size, hidden_size = config.hidden_size)
        self.sample_decoder = SampleDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.node_classifier = NodeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.mseLoss = nn.MSELoss()  # pytorch's mean squared error loss
        self.creLoss = nn.CrossEntropyLoss()  # pytorch's cross entropy loss (NOTE: no softmax is needed before)

    def boxDecoder(self, feature):
        return self.box_decoder(feature)

    def adjDecoder(self, feature):
        return self.adj_decoder(feature)

    def symDecoder(self, feature):
        return self.sym_decoder(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def boxLossEstimator(self, box_feature, gt_box_feature):
        return torch.cat([self.mseLoss(b, gt).mul(0.4) for b, gt in zip(box_feature, gt_box_feature)], 0)

    def symLossEstimator(self, sym_param, gt_sym_param):
        return torch.cat([self.mseLoss(s, gt).mul(0.5) for s, gt in zip(sym_param, gt_sym_param)], 0)

    def classifyLossEstimator(self, label_vector, gt_label_vector):
        return torch.cat([self.creLoss(l.unsqueeze(0), gt).mul(0.2) for l, gt in zip(label_vector, gt_label_vector)], 0)

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)


def decode_structure_fold(fold, feature, tree):
    
    def decode_node_box(node, feature):
        if node.is_leaf():
            box = fold.add('boxDecoder', feature)
            recon_loss = fold.add('boxLossEstimator', box, node.box)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            return fold.add('vectorAdder', recon_loss, label_loss)
        elif node.is_adj():
            left, right = fold.add('adjDecoder', feature).split(2)
            left_loss = decode_node_box(node.left, left)
            right_loss = decode_node_box(node.right, right)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return fold.add('vectorAdder', loss, label_loss)
        elif node.is_sym():
            sym_gen, sym_param = fold.add('symDecoder', feature).split(2)
            sym_param_loss = fold.add('symLossEstimator', sym_param, node.sym)
            sym_gen_loss = decode_node_box(node.left, sym_gen)
            label = fold.add('nodeClassifier', feature)
            label_loss = fold.add('classifyLossEstimator', label, node.label)
            loss = fold.add('vectorAdder', sym_gen_loss, sym_param_loss)
            return fold.add('vectorAdder', loss, label_loss)

    feature = fold.add('sampleDecoder', feature)
    loss = decode_node_box(tree.root, feature)
    return loss


#########################################################################################
## Functions for model testing: Decode a root code into a tree structure of boxes
#########################################################################################

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    return m

def decode_structure(model, root_code):
    """
    Decode a root code into a tree structure of boxes
    """
    decode = model.sampleDecoder(root_code)
    syms = [torch.ones(8).mul(10).cuda()]
    stack = [decode]
    boxes = []
    while len(stack) > 0:
        f = stack.pop()
        label_prob = model.nodeClassifier(f)
        _, label = torch.max(label_prob, 1)
        label = label.data
        if label[0] == 1:  # ADJ
            left, right = model.adjDecoder(f)
            stack.append(left)
            stack.append(right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if label[0] == 2:  # SYM
            left, s = model.symDecoder(f)
            s = s.squeeze(0)
            stack.append(left)
            syms.pop()
            syms.append(s.data)
        if label[0] == 0:  # BOX
            reBox = model.boxDecoder(f)
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