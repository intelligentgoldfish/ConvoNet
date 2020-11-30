%%applyErrorAlpha.m
%%Take weight/bias error and actually apply it
%%Because I'm too lazy to preallocate I'm going to try and edit everything
%%in place here

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

%%Current optimizer: SGD w/momentum

function [newW,newB] = sgd_momentum(w,b,dW,dB,learningRate)

num_layers = max(size(w));

newW = cell(size(w));
newB = cell(size(b));

% for n = 1:num_layers
%     kW = w{n};
%     kB = b{n};
%     aW = -1 .* (learningRate .* dW{n});
%     aB = -1 .* (learningRate .* dB{n});
%     
%     newW{n} = kW + aW;
%     newB{n} = kB + aB;
% end

for n = 1:num_layers
    kW = w{n};
    kB = b{n};
    aW = -1 .* (learningRate .* dW{n});
    aB = -1 .* (learningRate .* dB{n});
    
    newW{n} = kW + aW;
    newB{n} = kB + aB;
end

end