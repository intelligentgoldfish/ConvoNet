%%ReLu.m
%%Rectified Linear Activation Unit
%%Author: Thomas DeWitt

function z = ReLu(y)

n = size(y,1);
k = size(y,2);
j = size(y,3);

z = zeros(n,k,j);

for a = 1:n
    for b = 1:k
        for c = 1:j
            if y(a,b,c) <= 0
                z(a,b,c) = 0;
            else
                z(a,b,c) = y(a,b,c);
            end
        end
    end
end

end