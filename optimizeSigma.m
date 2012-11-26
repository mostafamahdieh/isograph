function sigma = optimizeSigma(ml, W, labels, li, eps, regress)

% Dt_pca : Data Set : each row is a data point
% labels : all labels : range from 1 to m
% li : index of the labeled data
% eps : gamma : regularization coefficient
% k1 : k in k-NN

minVal = ml(10^-1, W, labels, li, eps, regress);
%minVal

for i = 0:10
    val = ml(10^i, W, labels, li, eps, regress);
    %val

    if (val <= minVal)
        minVal = val;
    else
        break;
    end
end

a = 10^(i-2);
b = 10^(i);

if (a < b)
    %options = optimset('Display', 'iter-detailed');
    options = optimset('Display', 'off', 'TolX', 1);
    sigma = fminbnd(@(sigma)ml(sigma, W, labels, li, eps, regress), a, b, options);
else
    fprintf(['Error a >= b with a = ' num2str(a) ' and b = ' num2str(b) '\n']);
    sigma = 10^(i-1);
end
