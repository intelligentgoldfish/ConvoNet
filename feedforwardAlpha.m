%%forwardpropAlpha.m
%%First version of convolutional network feedforward code
%%As with init, still clunky, not recommended unless familiar with the code
%%Non-dev user interface will follow after network development is complete

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

function [a,z,preflat_size] = feedforwardAlpha(input,...
num_conv_layers,...
filters_per_conv,...
conv_size,...
pad,...
dense_dims,...
w,...
b)

input_dims = size(input);
if max(size(input_dims)) < 2
    error('Invalid input size: input must be at least 2-dimensional.');
end

%we assume we're working with images as this is a convolutional network
%also we're STILL only doing square inputs
input_edge = max(input_dims);
%input_depth = min(input_dims);

total_layers = num_conv_layers + max(size(dense_dims) - 1);

a = cell(1,total_layers);
z = cell(1,total_layers);

a{1} = input;
z{1} = input;

%% convolutional layers

for conv = 2:num_conv_layers+1
    %conv2D autocomputes filters - will a predefine accelerate exec speed?
    %...currently undetermined...
    %num_filters = filters_per_conv(conv);
    convW = w{conv-1};
    convB = b{conv-1};
    [a{conv},z{conv}] = conv2D(convW,convB,a{conv-1},conv_size,pad);
end

%% transition and feed into dense layers

if pad == true
    out_edge = input_edge;
else
    out_edge = input_edge - (conv_size - 1) * num_conv_layers;
end

out_params = out_edge^2 * size(a{num_conv_layers+1},3); %calculate # of dense input neurons
    
%dense_dims = [out_params dense_dims]; 

numLayers = length(dense_dims);

%reshape to dense column for feedforward
preflat_size = size(a{num_conv_layers+1});
a{num_conv_layers+1} = reshape(a{num_conv_layers+1},out_params,1);

%using output from conv stack as "input" here
for n = 2:numLayers %begin at 2 as first layer only feeds information forward
    x = w{num_conv_layers+n-1} * a{num_conv_layers+n-1} + b{num_conv_layers+n-1};
    z{num_conv_layers+n} = x;
    a{num_conv_layers+n} = sigmoid(x);
end

%reshape to stacked filters for convolutional backprop
%a{num_conv_layers+1} = reshape(a{num_conv_layers+1},preflat_size);

end