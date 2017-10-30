require 'nn'
require 'image'
require 'optim'

logger = optim.Logger('Transfer.log') -- logger can be changed  
logger:setNames{'Trainset Error', 'Testset Error'}


dataset = torch.load('flowers.t7')
classes = torch.range(1,17):totable() --17 classes
labels = torch.range(1,17):view(17,1):expand(17,80)
print(dataset:size()) --each class has 80 images of 3x128x128
image.display(dataset:select(2,3))

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end



shuffledData, shuffledLabels = shuffle(dataset:view(-1,3,128,128), labels:contiguous():view(-1))

trainSize = 0.85 * shuffledData:size(1)
trainData, testData = unpack(shuffledData:split(trainSize, 1))
trainLabels, testLabels = unpack(shuffledLabels:split(trainSize, 1))

print(trainData:size())

trainData = trainData:float() -- convert the data from a ByteTensor to a float Tensor.
trainLabels = trainLabels:float()

mean, std = trainData:mean(), trainData:std()
print(mean, std)
trainData:add(-mean):div(std)
    
testData = testData:float()
testLabels = testLabels:float()
testData:add(-mean):div(std)


-- Load GoogLeNet
googLeNet = torch.load('GoogLeNet_v2_nn.t7')

-- The new network
model = nn.Sequential()

for i=1,10 do
    local layer = googLeNet:get(i):clone()
    layer.parameters = function() return {} end --disable parameters
    layer.accGradParamters = nil --remove accGradParamters
    model:add(layer)
end

-- Check output dimensions with random input
model:float()
local y = model:forward(torch.rand(1,3,128,128):float())
print(y:size())

-- Add the new layers

