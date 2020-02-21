%%initConvNet.m
%%Initializes smoothed weights and biases for convolutional net
%%Utilizes the ConR framework

%%Version 1.0
%%Compatibility: Alpha Frame
%%Author: Thomas DeWitt

%%FIX PROBLEM OF INSERTED MAX-POOLING LAYERS INTERFERING WITH WEIGHT/BIAS
%%DISTRIBUTION


function [w,b] = initConvNet(conRLength,filters,filterSize,FClayerSize)

rng('shufffle'); %ensure that networks aren't double-generated

w = cell(1,3*conRLength+2); %implement ConR
b = cell(1,3*conRLength+2); %implement ConR

for n = 1:3*conRLength
    w{n} = randn(filterSize,filterSize,filters)./sqrt(conRLength); %weight init
    b{n} = randn(1,1,filters); %bias init
end

%compute weights/biases for fully connected layers as convolutions
for n = 1:2
    