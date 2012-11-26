clear;
runNum = 1;
% edgeNum = 290; % for usps
edgeNum = 300; % for mnist
% crit = zeros(runNum, edgeNum+1);
step = 50; % for mnist
% step = 10; % for usps
n = 1000; % sample instance size
basepath = sprintf('./java/GNN/gnn/%d', n);
k2 = 20;
k = 10;
seed = 1390;

% sigma1 = 634; % for usps
sigma1 = 806; % for mnist
for dataset = ['m']
		LoadDataset();
		N = size(Dt,1);  % number of instances
		labels = labels + (1-min(labels)); % normalize the labels to 1,2,...
		classN = max(labels);
		cvFolds = round(n/(classN*10)); % for both
		labeledNum = floor(n / cvFolds);
		RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
		rndTotal = randperm(N);
		DtSubset = Dt(rndTotal(1:n), :);
		labelsSubset = labels(rndTotal(1:n), :);

		varname = sprintf('%s_%d_%d_%d', dsname, n, k2, runNum);
		disp(['Loading graph ' varname]);
		filename = sprintf('%s/%s.txt', basepath, varname);
		load(sprintf(filename));
		WG = spconvert(eval(varname));
		disp('done');

		disp(['Loading Isograph ' varname]);
		load(sprintf('%s/Isograph_%s_%d_%d_%d.mat', basepath, dsname, n, k, 1));
		disp('done');

		disp(['Loading graph Betweenness ' varname]);
		betweennessFileID = fopen(sprintf('Betweenness_%s.txt', dsname), 'r');
		fscanf(betweennessFileID, '%s %d', 2);
		[BW ~] = fscanf(betweennessFileID, '%d %d %d');
		fclose(betweennessFileID);

		[ss ff ~] = find(abs(WI-W) > 0.01);
		r = zeros(length(ss));
		for i=1:length(ss)
			r(i) = WI(ss(i),ff(i)) / W(ss(i),ff(i));
		end
		[~, ind] = sort(r, 'descend');
		WG2 = WG > 0;
		
		per = randperm(n);
		
		WB = W;
		WS = W;
		
		p = 1;
		for i=0:step:edgeNum
			if (i >= 1)
				for j=0:step-1
					ep = i-step+j+1;
					WB(BW(3*ep-2),BW(3*(ep)-1)) = 0;
					WB(BW(3*ep-1),BW(3*(ep)-2)) = 0;
					WS(ss(ind(ep)),ff(ind(ep))) = 0;
					WS(ff(ind(ep)),ss(ind(ep))) = 0;
				end
			end
			W01B = WB > 0;
			W01S = WS > 0;
			L01B = (sparse(diag(sum(W01B))) - W01B);
			L01S = (sparse(diag(sum(W01S))) - W01S);
			for iter=1:cvFolds
				rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
				disp(['iteration ' num2str(i) ' ' num2str(iter)]);
				[s f w] = find(WS(rnd,rnd));
				WMLS = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
				LMLS = (sparse(diag(sum(WMLS))) - WMLS);
				
				[s f w] = find(WB(rnd,rnd));
				WMLB = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
				LMLB = (sparse(diag(sum(WMLB))) - WMLB);
				
				acc01S(iter, p) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01S(rnd,rnd), 0);
				accMLS(iter, p) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLS, 0);
				acc01B(iter, p) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01B(rnd,rnd), 0);
				accMLB(iter, p) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLB, 0);
			end
			p = p+1;
		end
		X1 = 0:step:edgeNum;
		YMatrix1 = [mean(acc01S(:, 1:p-1),1); mean(acc01B(:, 1:p-1),1); mean(accMLS(:, 1:p-1),1); mean(accMLB(:, 1:p-1),1)];
		figure1 = figure;
		% Create axes
		axes1 = axes('Parent',figure1);
		box(axes1,'on');
		hold(axes1,'all');

		% Create multiple lines using matrix input to plot
		plot1 = plot(X1,YMatrix1,'Parent',axes1);
		set(plot1(1),'Marker','^','Color',[1 0 0],'DisplayName','0-1 Isograph');
		set(plot1(2),'Marker','v','Color',[0 0 1],'DisplayName','0-1 EBC');
		set(plot1(3),'Marker','+','Color',[1 0 0],...
			'DisplayName','Weighted Isograph');
		set(plot1(4),'Marker','*','Color',[0 0 1],...
			'DisplayName','Weighted EBC');

		% Create xlabel
		xlabel('# of edges removed');

		% Create ylabel
		ylabel('Classification error (%)');

		% Create legend
		title({dsname});
		legend1 = legend(axes1,'show');
		set(legend1,...
			'Position',[0.451802817655622 0.540745386345359 0.413965087281795 0.258414766558089]);
end