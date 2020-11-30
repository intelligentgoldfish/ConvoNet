%%ReLuPrime
%%Derivative of ReLu function

function y = ReLuPrime(z)

n = size(z,1);
k = size(z,2);
j = size(z,3);

y = zeros(n,k,j);

for a = 1:n
    for b = 1:k
        for c = 1:j
            if z(a,b,c) <= 0
                y(a,b,c) = 1;
            else
                y(a,b,c) = 0;
            end
        end
    end
end


end