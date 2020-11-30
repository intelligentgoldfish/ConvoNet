%%backConv2D
%%"Full" convolution operation
%%For use in backpropagation ONLY

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

function gradient = backConv2D(w,delta,conv_size,prev_filters)

border = (conv_size-1);

init_edge = size(delta,1);

new_size = init_edge + border;

gradient = zeros(new_size,new_size,prev_filters);

filters = size(w,4);

strides = init_edge;

w = rot90(w,2);

delta_p = zeros(new_size+border,new_size+border,filters);
delta_p(1+border:init_edge+border,1+border:init_edge+border,:) = delta; %pad

for a = 1:filters
    for f = 1:prev_filters %through old filters
        for k = 1:strides %across
            for n = 1:strides %down
                slice = delta_p(n:n+conv_size-1, k:k+conv_size-1, f); %grab input
                slice = slice .* w(:, :, f, a); %multiply weight matrix
                out = sum(slice, [1 2 3]); %sum slice to complete convolution
                gradient(n, k, f) = gradient(n, k, f) + out; %store
            end
        end
    end
end

end