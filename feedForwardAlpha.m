%%feedForwardAlpha.m
%%Initial feedforward code for a convolutional neural network
%%Version 0.1
%%Author: Thomas DeWitt

%%CODE STRUCTURE
%-critical variable: dim
%-dim contains both the number and size of each convolutional and
%max-pooling layer
%-Three elements in dim are fixed: first layer (input), last layer
%(softmax), and second-to-last (fully-connected sigmoid)
%-All other layers alternate between convolutional and pooling

function [a,z,iA,iZ] = feedForwardAlpha(input,w,b,dim,numFeatureMaps,poolSizes)

numLayers = length(dim);

if rem(numLayers-1,2) ~= 0
    error('Invalid network dimensions.  Check for an invalid number of pooling or convolutional layers.');
end
if numLayers-3 < 2
    error('Not enough convolutional or pooling layers.');
end

a = cell(numFeatureMaps,numLayers);
z = cell(numFeatureMaps,numLayers);

features = numFeatureMaps;

for n = 1:features %feed input into network
    a{n,1} = input;
    z{n,1} = input;
end

for n = 1:features %perform convolutions/poolings sequentially per feature
    for k = 2:2:numLayers-2
        %perform convolution operation
        [a{n,k},z{n,k}] = featureMap(a{n,k-1},w{n,k-1},b{n,k-1},poolSizes{n,k-1});
        %perform max-pooling operation
        [a{n,k+1},z{n,k+1}] = maxPool(a{n,k},poolSizes{k});
    end
end

poolOutSize = size(a{1,numLayers-2});

%determine length of output vector of feature combination
fullyConnectedDim = poolOutSize(1) * poolOutSize(2) * features;

featuresOut = zeros(poolOutSize(1)*20,poolOutSize(2));

for n = 1:features
    featuresOut(20*n-19:20*n,:) = a{n,numLayers-2};
end

%reshape for fully connected layer
featuresOut = reshape(featuresOut',fullyConnectedDim,1); %note transpose

%retrieve fully connected weights and biases
iW{1} = w{1,size(w,2)-1};
iB{1} = b{1,size(b,2)-1};
iW{2} = w{1,size(w,2)};
iB{2} = b{1,size(b,2)};

%fully connected layer (2 sigmoid operations)
[iA,iZ] = feedForward3(featuresOut,iW,iB,[fullyConnectedDim 192 10]);

end




