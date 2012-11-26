function [acc] = laplacianLS(labels,labeledN,sparseL,regress)
%disp(['run lRate=' num2str(lRate) ' k= ' num2str(k) ' pc = ' num2str(pc) ' k2 = ' num2str(k2)]);
gamma = .02;
n = size(labels,1);
Il = sparse(1:n,1:n,1:n <= labeledN,n,n,n);

if (regress == 0)
	classN = max(labels);
	res_L = zeros(n, classN);
	%% Classification one-against-all
	%disp('Classification ...'); 
	for i = 1:classN
		% disp(['loop ' int2str(i) ' ...']);
		y = double(labels == i);
		y(~y) = -1;
		y(labeledN+1:n) = 0;
		%size(L)
		res_L(:, i) = (Il + gamma * sparseL) \ y;
	end
	[~, Out_L] = max(res_L');
	acc = (sum(Out_L(labeledN+1:n)'==labels(labeledN+1:n))/(n-labeledN)) * 100;
	% disp(['accuracy of Laplacian = %' num2str(acc)]);
	% disp('-------------------------------------------------------------')
else
	if (regress == 1)
		y = labels;
		y(labeledN+1:n, :) = 0;
		res_L = (Il + gamma * sparseL) \ y;
		% this is really error
		acc = mae(res_L(labeledN+1:n)-labels(labeledN+1:n));
	else
		y = labels;
		y(labeledN+1:n, :) = 0;
		res_L = (Il + gamma * sparseL) \ y;
		% this is really error
		acc = angle_error(res_L(labeledN+1:n,:),labels(labeledN+1:n,:));
	end
end