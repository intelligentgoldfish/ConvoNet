%%assembleAlpha.m
%%New network compile and train for Alpha Frame

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

clear; clc;

disp('~~~AlphaConvNet~~~');
disp('Author: Thomas DeWitt');
disp('Version 1.0.0');
disp(['This is a fully operational convolutional neural network.',newline,...
    'It has been entirely handwritten by the author.',newline,...
    'It will be very, VERY slow compared to Keras, Torch, etc.',newline,...
    'However, if you''re here, you''re not worried about speed.',newline,newline,...
    'Press any key to create and train a new network.',newline]);

pause;

disp('Initializing...');

% TRAINING HYPERPARAMETERS
epochs = 10;
learningRate = 0.01;
miniBatchSize = 10;
%mu = 0.9; %SGD w/momentum, coefficient of momentum

% INPUT DATA INFORMATION
input_edge = 28;
input_depth = 1;

% NETWORK DIMENSIONAL INFORMATION
num_conv_layers = 3;
filters_per_conv = [4 8 16];
conv_size = 3;
pad = false;
dense_dims = [32 10];


[w,b,params,dense_dims] = initAlpha(input_edge,...
                                    input_depth,...
                                    num_conv_layers,...
                                    filters_per_conv,...
                                    conv_size,...
                                    pad,...
                                    dense_dims);
                                                            
disp(['(i.e. we''re tuning a ',num2str(params),'-D function.)',newline]);
% disp(['Initializing parameter velocity...',newline]);
% weight_v = cell(1,length(b));
% bias_v = cell(1,length(b));

%initialize parameter velocity to 0 for SGD w/momentum
% for e = 1:length(b)
%     weight_v{e} = 0;
%     bias_v{e} = 0;
% end

disp(['Network initialized.',newline,'Training...']);

[trainW,trainB,accuracy] = trainAlpha(epochs,... %training hyperparameters
    miniBatchSize,...
    learningRate,...
    num_conv_layers,... %network parameters
    filters_per_conv,...
    conv_size,...
    pad,...
    dense_dims,...
    w,...
    b);

disp([newline,'Training complete.']);