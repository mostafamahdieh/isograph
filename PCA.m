n = 1000;
runNum = 10;
basepath = sprintf('java/GNN/gnn/%d', n);

for dataset = ['m']
	first = 1;
	seed = 1390;
	RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
	for iterNum=1:runNum
		disp([' iterNum=' num2str(iterNum)]);
		distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
		pc = 35; % number of pca dimensions
		LoadDataset();

		N = size(Dt,1);  % number of instances
		rndTotal = randperm(N);
		DtSubset = Dt(rndTotal(1:n), :);
		
		pc = min(pc, size(DtSubset,2));
		%% PCA
		disp('Principal Component Analysis ...');
		DtSubset = DtSubset - repmat(mean(DtSubset), n, 1);
		S = DtSubset' * DtSubset;
		[E, V] = eig(S);
		[~, Vsi] = sort(diag(V), 'descend');
		E = E(:, Vsi(1:pc));
		Dt = Dt * E;
		disp('-----------------------------------')
		           
		name = sprintf('%s/%s_PCA_Dt_%d_%d.txt', basepath, dsname, pc, iterNum);
		fileID = fopen(name, 'w');
		if (fileID < 0)
			disp(['error open file ' name]);
			return;
		end
		fprintf(fileID, '%d %d\r\n', N, pc);
		for i=1:N
			for j=1:pc
				if (j ~= 1)
					fprintf(fileID, ' ');
				end
				fprintf(fileID, '%f', Dt(i,j));
			end
			fprintf(fileID, '\r\n');
		end
		randperm(n);
		fclose(fileID);
	end
end