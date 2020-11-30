%%backpropAlpha.m
%%First version of convolutional backpropagation code
%%NOT recommended for anyone not extremely familiar with this code
%%If you intend to modify this code, please create a backup copy first
%%If this code fails to work after modification, you're on your own

%%Version 1.0.2
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

%%DISCLAIMER:  **ANY** MODIFICATION MADE TO THIS CODE WILL IMMEDIATELY
%%INVALIDATE ANY WRITTEN OR UNWRITTEN PROMISE MADE TO MAINTAIN AND SUPPORT
%%CODE IN OPERATIONAL ORDER

function [dW, dB, loss] = backpropAlpha(a,...
    z,...
    w,...
    b,...
    label,...
    num_conv_layers,...
    conv_size,...
    dense_dims,...
    preflat_size)

%ensure data has been correctly formatted
if size(label, 2) ~= 1 || size(label,1) ~= dense_dims(end)
    error('Desired output is not correct size for this network.  Check data formatting.');
end

numDenseLayers = length(dense_dims);

total_layers = num_conv_layers + numDenseLayers - 1;

gradient = (a{total_layers+1} - label); %gradient

loss = sum(gradient.^2)/2; %quadratic loss function

delta = gradient; %cross-entropy loss

for n = (total_layers):-1:total_layers-numDenseLayers+2 %for each dense layer
    dW{n} = delta * a{n}'; %calc and store weight error
    dB{n} = delta; %store bias error
    delta = w{n}' * delta .* reshape(sigmoidPrime(z{n}),[],1); %calc error in next layer
end

%%now to the interesting part: reshaping the gradient for convolutional
%%backpropagation...

delta = reshape(delta,preflat_size); %so this works...
a{num_conv_layers+1} = reshape(a{num_conv_layers+1},preflat_size);

%Backpropagation through a convolution is complex because of the janky
%calculus, but also because we have to downscale filters every time.  This
%makes it absolutely necessary that we carefully account for every single
%input channel in the layer before (i.e. layer n-1) and not create an error
%matrix larger than the input, but also requires us to account for each
%filter in the current layer (i.e. layer n).  This requires dual loops and
%extremely careful array indexing.

% THE FOLLOWING SECTION HAS BEEN EXTREMELY CAREFULLY ENGINEERED.  ANY
% MODIFICATION IS LIKELY TO RENDER IT NON-FUNCTIONAL, EVEN PERCEIVED
% ATTEMPTS TO ACCELERATE PERFORMANCE.  MODIFY AT YOUR OWN RISK, AND PLEASE
% ENSURE YOU HAVE BACKED THIS FILE UP

for n = num_conv_layers+1:-1:2 %stepping backwards through convolutional layers
    x = a{n}; %pull output
    wN = w{n-1}; %pull layer weights
    bN = b{n-1}; %pull layer biases
    wE = zeros(size(wN)); %allocate weight error
    filters = length(bN); %pull filters in layer
    prev_filters = size(wE,3); %pull input channels
    for f = 1:filters %iterating through filters
        for c = 1:prev_filters %iterating through channels
            wE(:,:,c,f) = conv2D(delta(:,:,f),0,x(:,:,c),size(delta,1),false); %remember w is a 4-D matrix
        end
    end
    bE = sum(delta,[1 2]); %calc bias error
    bE = reshape(bE,1,[]);
    dW{n-1} = wE; %store weight error
    dB{n-1} = bE; %store bias error
    delta = backConv2D(wN,delta,conv_size,prev_filters); %calculate new delta (dy, i.e. error in input)
end

end