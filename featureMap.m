%%featureMap.m
%%Pools the local receptive fields across an image for a single feature map
%%Takes only the weights and biases for hidden layers, as format is
%%currently indeterminate for deep CNN w/b format

%%Version 1.0.7
%%Author: Thomas DeWitt

function [a,z] = featureMap(input,w,b,poolSize)

%w is a 5x5 matrix applied across the entire input
%b is a 1x1 matrix applied for each sum

if poolSize ~= size(w,1) %throw error if weight matrix is too big or small
    error('Sizes of input matrix and weight matrix are incompatible.');
end

if max(size(b)) ~= 1
    error('Feature map bias must be a scalar.');
end

z = zeros(iterations);

for n = 1:iterations %move pooling down input layer
    for k = 1:iterations %move pooling across
        feature = input(n:n+4,k:k+4);
        weightedFeature = feature .* w;
        featureSum = sum(weightedFeature,[1 2]); %sum entire feature for neuron
        z(n,k) = featureSum + b;
    end
end

a = ReLu(z);

end