clear;

runNum = 10;
n = 1000; % sample instance size
basepath = sprintf('D:/GraphConstruction/Isograph/java/GNN/gnn/%d', n);
k = 10;
step = 20;
cvFolds = 5;
points = 5;
labeledNum = floor(n / cvFolds);

for dataset = ['m']
	LoadDataset();
	N = size(Dt,1);
	seed = 1390;
	RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
	for runIter=1:runNum
		rndTotal = randperm(N);
		DtSubset = Dt(rndTotal(1:n), :);
		labelsSubset = labels(rndTotal(1:n), :);
		per = randperm(n);
		for ip=0:points
			if (ip == 0)				
				varname = sprintf('kNN_%s_%d_%d_%d', dsname, n, k, runIter);
				disp(['Loading graph ' varname]);
				filename = sprintf('%s/%s.mat', basepath, varname);
				load(sprintf(filename));
				W01G = W01;
			else
				varname = sprintf('%s_%d_%d_%d_%d', dsname, n, k, runIter, ip);
				disp(['Loading graph ' varname]);
				filename = sprintf('%s/%s.txt', basepath, varname);
				load(sprintf(filename));
				W01G = spconvert(eval(varname));
				W01G = W01G > 0;
				disp('done');
			end

			L01G = (sparse(diag(sum(W01G))) - W01G);
			for iter=1:cvFolds
				rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
				disp(['iteration ' num2str(ip) ' ' num2str(iter)]);
				acc01G((runIter-1)*cvFolds+iter, ip+1) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01G(rnd,rnd), 0);
			end
		end
	end

	X1(1) = 1000;
	X1(2:points+1) = floor((N/points) .* (1:points));
	YMatrix1 = [100-mean(acc01G,1)]; % usps
%	YMatrix1 = [100-mean(acc01G,1)]; % mnist

	figure1 = figure;

	% Create axes
	axes1 = axes('Parent',figure1);
	box(axes1,'on');
	hold(axes1,'all');

	% Create multiple lines using matrix input to plot
	plot1 = plot(X1,YMatrix1,'Parent',axes1);
	set(plot1(1),'Marker','*','Color',[1 0 0],'DisplayName','gNN');

	% Create xlabel
	xlabel({'Total number of points'});

	% Create ylabel
	ylabel({'Error %'});

	% Create title
	title({dsname});

	% Create legend
	legend1 = legend(axes1,'show');
	set(legend1,...
		'Position',[0.7 0.8 0.1 0.1]);

end