%%ReLuPrime
%%Derivative of ReLu function

function y = ReLuPrime(z)

if z > 0
    y = 1;
else
    y = 0;
end

end