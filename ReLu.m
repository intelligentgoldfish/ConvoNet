%%ReLu.m
%%Rectified Linear Activation Unit
%%Author: Thomas DeWitt

function z = ReLu(y)

n = size(y,1);
k = size(y,2);

z = zeros(n,k);

for a = 1:n
    for b = 1:k
        if y(a,b) <= 0
            z(a,b) = 0;
        else
            z(a,b) = y(a,b);
        end
    end
end

end