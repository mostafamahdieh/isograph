distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

out = fopen('pose_estimation_1.txt', 'w');

for setid = [9, 10, 33, 40, 63, 102, 125, 1401, 1417, 1424]
	load(sprintf('dtset_pose/%d.mat', setid));
	Dt = X;
	labels = Y;
	N = size(Dt,1);  % number of instances
	cvFolds = 5;
	iterNum = cvFolds;	% number of iterations to run the algorithm
	n = N;
	k = 4;
	labeledNum = floor(n / cvFolds);

	rndTotal = randperm(N);
	DtSubset = Dt(rndTotal(1:n), :);
	labelsSubset = labels(rndTotal(1:n), :);

	if (distype == 1)
		pc = min(pc, size(DtSubset,2));
		%% PCA
		disp('Principal Component Analysis ...');
		DtSubset = DtSubset - repmat(mean(DtSubset), n, 1);
		S = DtSubset' * DtSubset;
		[E, V] = eig(S);
		[~, Vsi] = sort(diag(V), 'descend');
		E = E(:, Vsi(1:pc));
		DtSubset = DtSubset * E;
		disp('-----------------------------------')
	end

	W = sparse(n,n);
	W01 = sparse(n,n);

	kl = 2;
	Wk = sparse(n,n);
	DtSubsetNorm = sqrt(sum(DtSubset .^ 2,2));

	for i = 1:n
		ithd = repmat(DtSubset(i,:), n, 1);
		ithdNorm = DtSubsetNorm(i);
		if (distype == 1)
			dis = sqrt(sum((ithd-DtSubset).^2, 2));
		else
			dis = 1.001-sum(ithd.*DtSubset,2) ./ (DtSubsetNorm * ithdNorm);
		end
		[disSorted, IX] = sort(dis);
		W(i, IX(2:k+1)) = disSorted(2:k+1);
		W(IX(2:k+1),i) = disSorted(2:k+1);
		W(i,i)=0;

		W01(i, IX(2:k+1)) = 1;
		W01(IX(2:k+1),i) = 1;
		W01(i,i) = 0;	

		Wk(i, IX(2:kl+1)) = 1;
		Wk(IX(2:kl+1),i) = 1;
		Wk(i,i) = 0;	
	end

	WI = IsographReweight(W, Wk, W > 0, n, 2.0, 3, 0.5, 2);
	W01I = W01;
	W01I(abs(WI-W) > 0.1) = 0;

	disp('Graph built');


	acc01 = zeros(iterNum, 1);
	accML = zeros(iterNum, 1);
	acc01I = zeros(iterNum, 1);
	accMLI = zeros(iterNum, 1);

	L01 = (sparse(diag(sum(W01))) - W01);
	L01I = (sparse(diag(sum(W01I))) - W01I);

	per = randperm(n);

	for iter=1:iterNum
		rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
% 		if(iter==1)
% 			sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, 1)
% 			sigma1 = abs(sigma1);
% 		end
% 		[s f w] = find(W(rnd,rnd));
% 		WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
% 		LML = (sparse(diag(sum(WML))) - WML);
% 
% 		if(iter==1)
% 			sigma2 = optimizeSigma(@ML, WI(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum,0.02, 1)
% 			sigma2 = abs(sigma2);
% 		end
% 		[s f w] = find(WI(rnd,rnd));
% 		WMLI = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
% 		LMLI = (sparse(diag(sum(WMLI))) - WMLI);

		acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd), 2);
%		accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML, 1);
		acc01I(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01I(rnd,rnd), 2);
%		accMLI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLI, 1);
	end
	
%	fprintf('%d acc01 mean=%f stddev=%f\n', setid, mean(acc01), std(acc01));
%	fprintf('%d acc01I mean=%f stddev=%f\n', setid, mean(acc01I), std(acc01I));
% 	fprintf('%d accML mean=%f stddev=%f\n', setid, mean(accML), std(accML));
% 	fprintf('%d accMLI mean=%f stddev=%f\n', setid, mean(accMLI), std(accMLI));
% 	accMLMean = mean(accML);
% 	accMLIMean = mean(accMLI);
	fprintf(out, '%d %f %f %f %f\n', setid, mean(acc01), std(acc01), mean(acc01I), std(acc01I));
end
fclose(out);
