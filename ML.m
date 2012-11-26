function [ml] = ML(sigma, W, labels, li, eps, regress)
% labels : all labels : range from 1 to m
% li : index of the labeled data
% eps : gamma : regularization coefficient
% k1 : k in k-NN

num = size(W, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[s, f, w] = find(W);
if (sigma == 0)
    Wp = sparse(s, f, w ~= 0, num, num);
else
    Wp = sparse(s, f, exp(-w.^2/(sigma*sigma)), num, num);
end

d = sum(Wp);
D = diag(d);
%D2 = diag(d.^-0.5);
%L = D2*(D - Wp)*D2;
L = (D - Wp);
%Lp = pinv(L);

[E, V] = eig(full(L));
v = diag(V);
[vs, vsi] = sort(diag(V),'descend');

for i = length(vs) : -1 : 1
    if (vs(i) > 0)
        break;
    end
end
i = length(vs) - 1;
Lp = E(:, vsi(1:i)) * diag(1./v(vsi(1:i))) * E(:, vsi(1:i))';

val = 0;
%eps = 0.01;

warning off all
X = inv(Lp(li, li) + eps * eye(length(li)));
warning on all

if (regress == 0)
	nfe = max(labels);
	for i = 1 : nfe
		y = labels(li) == i;
		y = 2 * y - 1;
		val = val + 0.5 * y' * X * y + 0.5 * log(det(Lp(li, li) + eps * eye(length(li))));
	end
else
	for k=1:size(labels,2)
		y = labels(li,k);
		val = val + 0.5 * y' * X * y;
	end
	val = val + (0.5 * log(det(Lp(li, li) + eps * eye(length(li))))) * size(labels,2);
end

if (norm(val) == inf || isnan(norm(val)))
    val = 10^30;
end
ml = val;
%mlg = MLgrad(sigma, Dt, labels, li, eps, k);
%mlg = mlg';