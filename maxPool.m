%%maxPool.m
%%Max-pooling layer for a convolutional neural network

%%Version 1.0.0
%%Author: Thomas DeWitt

function maxPooledLayer = maxPool(input,poolSize)

inputSize = size(input,1);

if rem(inputSize,poolSize) ~= 0 %throw error if not divisible
    error('Convolutional layer and pooling size are incompatible.');
end

poolEdge = inputSize/poolSize;

maxPooledLayer = zeros(poolEdge);

for n = 1:poolEdge
    for k = 1:poolEdge
        pool = input(2*n-poolSize+1:2*n,2*k-poolSize+1:2*k);
        poolMax = max(pool,[],[1 2]);
        maxPooledLayer(n,k) = poolMax;
    end
end

end