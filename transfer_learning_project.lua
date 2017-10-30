require 'nn'
require 'image'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

local NumClasses = 4
--local NumClasses = 8
--local NumClasses = 12
--local NumClasses = 16

logger = optim.Logger('Transfer.log') -- logger can be changed  
logger:setNames{'Trainset Error', 'Testset Error'}


dataset = torch.load('flowers.t7'):narrow(1,1,NumClasses)
classes = torch.range(1,17):totable() --17 classes
labels = torch.range(1,17):view(17,1):expand(17,80):narrow(1,1,NumClasses)
print(labels:size())
print(dataset:size()) --each class has 80 images of 3x128x128
image.display(dataset:select(2,1))

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end


shuffledData, shuffledLabels = shuffle(dataset:view(-1,3,128,128), labels:contiguous():view(-1))

trainSize = 0.85 * shuffledData:size(1)
trainData, testData = unpack(shuffledData:split(trainSize, 1))
trainLabels, testLabels = unpack(shuffledLabels:split(trainSize, 1))

print(testData:size(1))

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
print("output dimensions:")
print(y:size())

-- Add the new layers
model:add(nn.SpatialConvolution(320, 16, 3, 3)) -- input dimension (320 input image channels), output dimension (creates 16 feature maps(kernels)), 3x3 kernels
model:add(nn.ReLU()) -- activation layer
model:add(nn.SpatialMaxPooling(4,4,4,4)) -- 4x4 windows
model:add(nn.View(16*3*3)) -- reshape image size to vector 
model:add(nn.Dropout())
model:add(nn.Linear(16*3*3, NumClasses)) 
model:add(nn.LogSoftMax()) -- translates output to log probabilities
print(tostring(model))

-- Loss Function = Negative Log Likelihood ()
lossFunc = nn.ClassNLLCriterion():float() 
w, dE_dw = model:getParameters()

print('Number of parameters:', w:nElement())

batchSize = 32 -- number of examples in each step
epochs = 200
optimState = {
    --learningRate = 0.01,
    
}

function forwardNet(data, labels, train)
    local confusion = optim.ConfusionMatrix(torch.range(1,NumClasses):totable()) -- create confusion matrix (numClasses size)
    if train then
        --set network into training mode
        model:training()
    end

    local x = data:narrow(1, 1, batchSize):float() -- put in x only batchSize examples from set
    local yt = labels:narrow(1, 1, batchSize):float() -- same with labels
    local y = model:forward(x) -- put in y the result of the model with x (log of probabilty)
    local err = lossFunc:forward(y, yt) -- use loss function to compare the original size to the new y size and calculate the loss
    confusion:batchAdd(y,yt)

        
    if train then
        function feval()
            model:zeroGradParameters() --zero grads
            local dE_dy = lossFunc:backward(y,yt)
            model:backward(x, dE_dy) -- backpropagation (calculate fix for the parameters)
            
            return err, dE_dw
        end
            
        optim.adam(feval, w, optimState)
    end
    
    confusion:updateValids()
    local avgLoss = err
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    if testError[e] < 0.1 then
    	break
--  if testError[e] < 0.15 then
--  	break
--  if testError[e] < 0.2 then
--    	break
   	end
    logger:add{trainError[e],testError[e]} -- loss is the value which you want to plot
    logger:style{'-','-'}   -- the style of your line, as in MATLAB, we use '-' or '|' etc.

    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)

    end
end

for i=1 ,10 do
	testData, testLabels = shuffle(testData, testLabels)
	img=testData:narrow(1,1,1)
	image.display(img)
	x = model:forward(testData:narrow(1,1,1):float())
	x:exp()
	print("Picture number",i)
	print(x)
end

logger:plot()
