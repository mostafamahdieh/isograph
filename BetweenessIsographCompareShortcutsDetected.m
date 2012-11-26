runNum = 1;
% edgeNum = 99; % mnist
edgeNum = 290; % usps
accI = zeros(runNum, edgeNum+1);
accB = zeros(runNum, edgeNum+1);
crit = zeros(runNum, edgeNum+1);
n = 1000; % sample instance size
basepath = sprintf('D:/GraphConstruction/Isograph/java/GNN/gnn/%d', n);
k2 = 20;
k = 10;
step = 2;

for dataset = ['m']
	switch dataset
		case 'm'
			dsname = 'MNIST';
		case 'u'
			dsname = 'USPS';
	end
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

	[s f ~] = find(abs(WI-W) > 0.1);
	r = zeros(length(s));
	for i=1:length(s)
%		r(i) = WI(s(i),f(i)) / W(s(i),f(i));
		r(i) = WI(s(i),f(i)) / W(s(i),f(i));
	end
%	[~, ind] = sort(r, 'descend');
	[~, ind] = sort(r, 'descend');

	accI = zeros(1, edgeNum+1);
	accB = zeros(1, edgeNum+1);
	crit = zeros(1, edgeNum+1);

	WG2 = WG > 0;

	for i=1:edgeNum
		accI(i+1) = accI(i) + (1-WG2(s(ind(i)),f(ind(i))));
		if (3*i <= length(BW))
			accB(i+1) = accB(i) + (1-WG2(BW(3*i-2),BW(3*i-1)));
			crit(i+1) = BW(3*i);
		else
			accB(i+1) = accB(i);
			crit(i+1) = crit(i);
		end
	end
	X1 = 1:step:edgeNum+1;
	YMatrix1 = [accI(1:step:edgeNum+1); accB(1:step:edgeNum+1);]; % crit(1:step:edgeNum+1)*0.01];

figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
box(axes1,'on');
hold(axes1,'all');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'Parent',axes1);
set(plot1(1),'Marker','*','Color',[1 0 0],'DisplayName','Isograph');
set(plot1(2),'Marker','+','Color',[0 0 1],'DisplayName','EBC');

% Create xlabel
xlabel({'# of edges removed'});

% Create ylabel
ylabel({'# of correctly detected shortcuts'});

% Create title
title({dsname});

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
	'Position',[0.140190972222221 0.69764957264957 0.278645833333333 0.194444444444444]);

end