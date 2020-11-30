%%trainAlpha.m
%%Training harness for Alpha Frame

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

function [trainW,trainB,accuracy] = trainAlpha(epochs,... %training hyperparameters
    miniBatchSize,...
    learningRate,...
    num_conv_layers,... %network parameters
    filters_per_conv,...
    conv_size,...
    pad,...
    dense_dims,...
    w,...
    b)

disp('Prepping data...');
[train,test] = loadDataAlpha();

rng('shuffle');
disp(['Done.',newline,'Commencing epochs...',newline]);

%numBatches = size(train,1)/miniBatchSize;
numBatches = 5;

disp('Starting timer...');
tic; %initialize chronometer

correct = 0;
for k = 1:size(test,1)/200
    output = identifyDigit(test{k,1},w,b,num_conv_layers,filters_per_conv,conv_size,pad,dense_dims);
    if output == test{k,2}
        correct = correct + 1;
    end
end
stat = 100*correct/k;

disp(string(stat))

for epoch = 1:epochs
    
    %shuffle data and maintain label order
    shuffleOrder = randperm(size(train,1));
    train = train(shuffleOrder,:);
    
    for m = 1:numBatches
        %pull data sequentially for each mini-batch
        batchStart = 1 + (m-1) * miniBatchSize; 
        batchRange = batchStart:(batchStart+miniBatchSize-1);
        thisBatch = train(batchRange,:);
        
        %backprop and tweak
        [dW,dB,loss] = processBatchAlpha(thisBatch,...
                                        num_conv_layers,...
                                        filters_per_conv,...
                                        conv_size,...
                                        pad,...
                                        dense_dims,...
                                        w,...
                                        b);
        [w,b] = sgd_momentum(w,b,dW,dB,learningRate);
        disp(['Batch ',num2str(m),' of ',num2str(numBatches),' complete.']);
    end
    
    correct = 0;
    for k = 1:size(test,1)/200
        output = identifyDigit(test{k,1},w,b,num_conv_layers,filters_per_conv,conv_size,pad,dense_dims);
        if output == test{k,2}
            correct = correct + 1;
        end
    end
    stat = 100*correct/k;
    
    timeElapsed = toc;

    %display progress
    disp(['Epoch ',num2str(epoch),' of ',num2str(epochs),' complete.']);
    disp(['Time elapsed: ',num2str(timeElapsed),' seconds.']);
    disp([num2str(stat),'% accuracy.']);
    disp(['Average loss: ',num2str(loss),newline]);
end

%avgAccuracy = 100 * avgAccuracy / epochs;

trainW = w;
trainB = b;

correct = 0;
for k = 1:size(test,1)
    output = identifyDigit(test{k,1},w,b,num_conv_layers,filters_per_conv,conv_size,pad,dense_dims);
    if output == test{k,2}
        correct = correct + 1;
    end
end
accuracy = 100*correct/size(test,1);

end