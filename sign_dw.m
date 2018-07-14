function output = sign_dw(input)

    output = ones(size(input));
    indices = find(input < 0);
    output(indices)=-1;
end