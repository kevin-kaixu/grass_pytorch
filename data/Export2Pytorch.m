clear all
trainingData = load('../data/trainingData_chair.mat');
data = trainingData.data;
dataNum = length(data);

maxBoxes = 30;
maxOps = 50;
maxSyms = 10;
maxDepth = 10;
copies = 1;

boxes = zeros(12, maxBoxes*dataNum*copies);
ops = zeros(maxOps,dataNum*copies);
syms = zeros(8,maxSyms*dataNum*copies);
weights = zeros(1,dataNum*copies);

for i = 1:dataNum
    p_index = i;
    
    symboxes = data{p_index}.symshapes;
    treekids = data{p_index}.treekids;
    symparams = data{p_index}.symparams;
    b = size(symboxes,2);
    l = size(treekids,1);
    tbox = zeros(12, b);
    top = -ones(1,l);
    tsym = zeros(8,1);
    box = zeros(12, maxBoxes);
    op = -ones(maxOps,1);
    sym = zeros(8,maxSyms);
    
    stack = [treekids(l, 1), treekids(l, 2)];
    top(1) = 1;
    count = 2;
    bcount = 1;
    scount = 1;
    
    while size(stack,2) ~= 0
        idx = size(stack,2);
        node = stack(idx);
        stack(idx) = [];
        left = treekids(node, 1);
        right = treekids(node, 2);
        if left == 0 && right == 0
            top(count) = 0;
            tbox(:, bcount) = symboxes(:, node);
            count = count + 1;
            bcount = bcount + 1;
            continue;
        end
        if left ~= 0 && right == 0
            top(count) = 2;
            stack(idx) = left;
            tsym(:,scount) = symparams{left};
            count = count + 1;
            scount = scount + 1;
            continue;
        end
        if left ~= 0 && right ~= 0
            top(count) = 1;
            stack(idx) = left;
            stack(idx+1) = right;
            count = count + 1;
            continue;
        end
    end
    top = fliplr(top);
    tsym = fliplr(tsym);
    tbox = fliplr(tbox);
    box(:, 1:b) = tbox;
    sym(:, 1:size(tsym,2)) = tsym;
    op(1:l, 1) = top';
    
    box = repmat(box, 1, copies);
    op = repmat(op, 1, copies);
    sym = repmat(sym, 1, copies);
    boxes(:, (i-1)*maxBoxes*copies+1:i*maxBoxes*copies) = box;
    ops(:,(i-1)*copies+1:i*copies) = op;
    syms(:, (i-1)*maxSyms*copies+1:i*maxSyms*copies) = sym;
    weights(:, (i-1)*copies+1:i*copies) = b/maxBoxes;
end

save('boxes.mat','boxes');
save('ops.mat','ops');
save('syms.mat','syms');
save('weights.mat','weights');