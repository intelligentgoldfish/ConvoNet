%%featureMap3.m
%%Revised feature mapping combining all feature mappings into single script
%%Combined both convolution and ReLU operations
%%Currently only scripted for square inputs to preserve my sanity
%%Does not currently independently use stride (i.e. stride = 1)

%%Version 3.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

% w: 4-D matrix, 4th dimension is filters
% b: 1-D vector, 1 entry per filter
% input: 3-D matrix
% convSize: length/width of square convolutions
% padding: 'valid' or 'same' for no padding/padding

function [a,z] = conv2D(w,b,inputMatrix,convSize,padding)

if convSize ~= size(w,1) %throw error if weight matrix is too big or small
    error('Filter size and weight matrix are incompatible.');
end

filters = size(w,4);

if max(size(b)) ~= filters || min(size(b)) ~= 1
    error('Feature map bias must be a fiber vector.');
end

strides = size(inputMatrix,1) - convSize + 1;

%rawOutputSize = size(iterations,1);

z = zeros(strides,strides,filters);
%aRaw = zeros(size(z));

for f = 1:filters %move through filters
    for k = 1:strides %move across input
        for n = 1:strides %move down input
            slice = inputMatrix(n:n+convSize-1, k:k+convSize-1, :); %grab input
            slice = slice .* w(:, :, :, f); %multiply weight matrix
            conv = sum(slice, [1 2 3]); %sum slice to complete convolution
            out = conv + b(f); %add bias
            z(n, k, f) = out; %store
        end
    end
end

aRaw = ReLu(z); %perform ReLu activation

if strcmp(padding,'same') == false
    a = aRaw;
else
    edgeLength = size(inputMatrix,1); %original input size
    a = zeros(edgeLength,edgeLength,filters); %0-matrix matching height, width of input
    border = (edgeLength - strides)/2; %zeros 'border' between input, output size
    a(1+border:edgeLength-border,1+border:edgeLength-border,filters) = aRaw; %pad output
end
    
%a = zeros((rawOutputSize + 2*padding),(rawOutputSize + 2*padding),filters);

end