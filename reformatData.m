%%reformatData.m
%%Reformats column vector as square matrix in correct image orientation

%%Version 1.0.0
%%Author: Thomas DeWitt

function squareData = reformatData(imageVector)

squareData = transpose(reshape(imageVector,28,28));

end