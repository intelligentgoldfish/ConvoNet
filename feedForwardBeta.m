%%feedForwardBeta.m
%%Beta edition of feedforward framework for convolutional network
%%Utilizes the ConR framework

%%Version 0.3
%%Compatibility: Alpha Frame
%%Author: Thomas DeWitt

function [a,z] = feedForwardBeta(input,w,b,conRLength,poolSize)

getSize = size(w{1});

filters = getSize(3); %get number of filters
filterSize = getSize(1); %get filter size

a = cell(1,3*conRLength+3);
z = cell(1,3*conRLength+3);

augmentedInput = input;

for n = 1:filters %build input up for different filters for ease of access
    augmentedInput = cat(augmentedInput,input,3); 
end

a{1} = input;
z{1} = input;

for n = 2:3:3*conRLength+1
    [a{n},z{n}] = featureMap2(w{n},b{n},a{n-1},filters,filterSize,~,1);
    [a{n+1},z{n+1}] = featureMap2(w{n+1},b{n+1},a{n},filters,filterSize,~,1);
    %%ADDRESS PROBLEM OF INSERTED MAX-POOLING INTERFERING WITH W/B INDEXING
    pooledInput = maxPool(a{n+1},poolSize);
    a{n+2} = pooledInput;
    z{n+2} = pooledInput;
end

%%insert FC-Conv layer here

%%ADDRESS DIFFICULTY OF CONVOLUTIONAL FULLY-CONNECTED LAYER

end



