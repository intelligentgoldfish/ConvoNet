%%featureMap2.m
%%Revised feature mapping combining all feature mappings into single script
%%Combined both convolution and ReLU operations
%%Currently only scripted for square inputs to preserve my sanity
%%Does not currently independently use stride (i.e. stride = 1)

%%Version 2.0
%%Author: Thomas DeWitt

function [a,z] = featureMap2(w,b,input,filters,filterSize,~,padding)

if filterSize ~= size(w,1) %throw error if weight matrix is too big or small
    error('Filter size and weight matrix are incompatible.');
end

if max(size(b)) ~= filters || min(size(b)) ~= 1
    error('Feature map bias must be a fiber vector.');
end

iterations = size(input,1) - filterSize + 1;

rawOutputSize = size(iterations,1);

z = zeros(rawOutputSize,rawOutputSize,filters);
a = zeros((rawOutputSize + 2*padding),(rawOutputSize + 2*padding),filters);

for f = 1:filters %move through filters and generate fiber
    for n = 1:iterations %move pooling down input layer
        for k = 1:iterations %move pooling across
            feature = input(n:n+filterSize-1,k:k+filterSize-1);
            weightedFeature = feature .* w(f); %elementwise weight multiplication
            featureSum = sum(weightedFeature,[1 2]); %sum entire feature for neuron
            z(n,k,f) = featureSum + b(f); %add bias
        end
    end
    a(padding+1:rawOutputSize+padding+1,padding+1:rawOutputSize+padding+1,f) = z(:,:,f);
end

end