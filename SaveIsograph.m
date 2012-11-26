n = 1000;
runNum = 1;
k = 10;
basepath = sprintf('D:/GraphConstruction/Isograph/gnn/%d', n);

for dataset = ['m']
	first = 1;
	seed = 1390;
	RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
	for iterNum=1:runNum
		disp(['Compare k=' num2str(k) ' iterNum=' num2str(iterNum)]);
		distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
		pc = 35; % number of pca dimensions
		LoadDataset();

		N = size(Dt,1);  % number of instances
		rndTotal = randperm(N);
		DtSubset = Dt(rndTotal(1:n), :);
		
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

		kl = 3;
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

			Wk(i, IX(2:kl+1)) = 1;
			Wk(IX(2:kl+1),i) = 1;
			Wk(i,i) = 0;	
		end

		disp('Running Isograph reweight');
		WI = IsographReweight(W, Wk, W > 0, n, 2.0, 3, 0.5, 2);
		save(sprintf('%s/Isograph_%s_%d_%d_%d.mat', basepath, dsname, n, k, iterNum), 'W', 'WI');
	end
end