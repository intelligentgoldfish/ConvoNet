%%processBatchAlpha.m
%%mini-batch processing

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

function [eW,eB,avg_loss] = processBatchAlpha(data,...
    num_conv_layers,...
    filters_per_conv,...
    conv_size,...
    pad,...
    dense_dims,...
    w,...
    b)


%% BEGIN EMTPY INIT
ex = data{1,1};
% input_edge = size(ex,1);
input_depth = size(ex,3);
ewSum = cell(1,num_conv_layers+max(size(dense_dims))-1); %weight cell
ebSum = cell(1,num_conv_layers+max(size(dense_dims))-1); %bias cell
depth = input_depth; %initial working input depth
for n = 1:num_conv_layers
    ewSum{n} = zeros(conv_size, conv_size, depth, filters_per_conv(n)); %init conv weights
    depth = filters_per_conv(n); %update depth for output dims of initialized layer
    ebSum{n} = zeros(1, depth); %init conv biases
end 
numLayers = length(dense_dims);
dim = dense_dims;
for n = 1:numLayers-1
    ewSum{num_conv_layers + n} = zeros(dim(n+1),dim(n))./sqrt(dim(n));   %init dense weights
    ebSum{num_conv_layers + n} = zeros(dim(n+1),1); %init dense biases
end
%% END EMPTY INIT


% ewSum = cell(1,max(size(w)));
% ebSum = cell(1,max(size(b)));
% 
% for e = 1:length(ewSum)
%     ewSum{e} = 0;
%     ebSum{e} = 0;
% end

eW = cell(1,max(size(w)));
eB = cell(1,max(size(b)));

batchSize = max(size(data));

numLayers = num_conv_layers + length(dense_dims) - 1;

total_loss = 0;

for n = 1:batchSize
    input = data{n,1};
    label = data{n,2};
    
    %feedforward
    [a,z,preflat_size] = feedforwardAlpha(input,...
    num_conv_layers,...
    filters_per_conv,...
    conv_size,...
    pad,...
    dense_dims,...
    w,...
    b);
    
    %backpropagate
    [dW, dB, loss] = backpropAlpha(a,...
    z,...
    w,...
    b,...
    label,...
    num_conv_layers,...
    conv_size,...
    dense_dims,...
    preflat_size);
    
    %add to sigma(dc/dw) and sigma(dc/db)
    for k = 1:numLayers
        ewSum{k} = ewSum{k} + dW{k};
        ebSum{k} = ebSum{k} + dB{k};
    end
    
    total_loss = total_loss + loss;
end 

for n = 1:numLayers
    eW{n} = ewSum{n}./batchSize;
    eB{n} = ebSum{n}./batchSize;
end

avg_loss = total_loss/batchSize;

end