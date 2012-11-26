function [acc01Mean accMLMean pc n] = GNNTikhonov(dataset,k,runNum,labeledPerClass)

distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

disp('Loading dataset');

switch dataset
	case 'm'
		dsname = 'MNIST';
		load 'dtset/MNIST';
	case 'u'
		dsname = 'USPS';
		load 'dtset/USPS';
end

disp('done');

N = size(Dt,1);  % number of instances
n = 1500;	 % sample instance size
labels = labels + (1-min(labels)); % normalize the labels to 1,2,...

classN = max(labels);
cvFolds = round(n/(classN*labeledPerClass));
iterNum = cvFolds;	% number of iterations to run the algorithm
labeledNum = floor(n / cvFolds);

rndTotal = randperm(N);
labelsSubset = labels(rndTotal(1:n), :);

acc01 = zeros(iterNum, 1);
accML = zeros(iterNum, 1);

varname = sprintf('%s_GNN_%d_%d', dsname, k, runNum);
filename = sprintf('%s.txt', varname);

disp(['Loading graph ' varname]);
load(sprintf(filename));
W = spconvert(eval(varname));
disp('done');

W01 = W > 0;
disp(['mean degree ' num2str(mean(sum(W01)))]);
L01 = (sparse(diag(sum(W01))) - W01);
per = randperm(n);

for iter=1:iterNum
	disp(['iteration ' num2str(iter)]);
	rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
% 	if(iter==1)
% 		disp('Running ML');
% 		sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd), 1:labeledNum, 0.02);
% 		sigma1 = abs(sigma1);
% 		disp('done');
% 	end
% 	
% 	[s f w] = find(W(rnd,rnd));
% 	WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
% 	LML = (sparse(diag(sum(WML))) - WML);

	acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd), 0);
% 	accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML);
end

fprintf('acc01 mean=%f stddev=%f\n', mean(acc01), std(acc01));
fprintf('accML mean=%f stddev=%f\n', mean(accML), std(accML));

acc01Mean = mean(acc01);
accMLMean = mean(accML);
end