clear;
n = 1000;
runNum = 10;
k = 10;
basepath = sprintf('D:/GraphConstruction/Isograph/java/GNN/gnn/%d', n);

for dataset = ['u' 'm']
	distype = 1;
	LoadDataset();
	first = 1;
	seed = 1390;
	RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
	pc = 35; % number of pca dimensions
	N = size(Dt,1);  % number of instances
	for iterNum=1:runNum
		disp(['Compare k=' num2str(k) ' iterNum=' num2str(iterNum)]);
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
		W01 = sparse(n,n);
		for i = 1:n
			ithd = repmat(DtSubset(i,:), n, 1);
			dis = sqrt(sum((ithd-DtSubset).^2, 2));
			[disSorted, IX] = sort(dis);
			W(i, IX(2:k+1)) = disSorted(2:k+1);
			W(IX(2:k+1),i) = disSorted(2:k+1);
			W(i,i)=0;

			W01(i, IX(2:k+1)) = 1;
			W01(IX(2:k+1),i) = 1;
			W01(i,i) = 0;	
		end
		
		randperm(n);

		varname = sprintf('%s_%d_%d_%d', dsname, n, k, iterNum);
		disp(['saving graph ' varname]);
		filename = sprintf('%s/kNN_%s.txt', basepath, varname);				
		save(filename, 'W', 'W01');
		filename = sprintf('%s/kNN_%s.mat', basepath, varname);		
		save(filename, 'W', 'W01');
	end
end