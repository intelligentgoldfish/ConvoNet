%%initAlpha.m
%%First version of convolutional network weight/bias initialization
%%Still quite clunky, not recommended for novice users
%%Interfaced initialization will take place later once network is done

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0


function [w,b,params,dense_dims] = initAlpha(input_edge,...
input_depth,...
num_conv_layers,...
filters_per_conv,...
conv_size,...
pad,...
dense_dims)


rng('shuffle');

params = 0;

w = cell(1,num_conv_layers+max(size(dense_dims))-1); %weight cell
b = cell(1,num_conv_layers+max(size(dense_dims))-1); %bias cell

depth = input_depth; %initial working input depth

for n = 1:num_conv_layers
    w{n} = randn(conv_size, conv_size, depth, filters_per_conv(n)); %init conv weights
    depth = filters_per_conv(n); %update depth for output dims of initialized layer
    b{n} = randn(1, depth); %init conv biases
    params = params + numel(w{n}) + numel(b{n}); %calc parameters added
end

if pad == true
    out_edge = input_edge;
else
    out_edge = input_edge - (conv_size - 1) * num_conv_layers;
end

out_params = out_edge^2 * depth; %calculate # of dense input neurons
    
dense_dims = [out_params dense_dims]; 

numLayers = length(dense_dims);

dim = dense_dims;

for n = 1:numLayers-1
    w{num_conv_layers + n} = randn(dim(n+1),dim(n))./sqrt(dim(n));   %init dense weights
    b{num_conv_layers + n} = randn(dim(n+1),1); %init dense biases
    params = params + numel(w{num_conv_layers + n}) + numel(b{num_conv_layers + n}); %calc params
end

disp(['Parameter count: ',num2str(params)]);

end