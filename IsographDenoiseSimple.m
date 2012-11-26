function [acc01Mean acc01DMean acc01DIMean accMLMean accMLDMean accMLDIMean] = IsographDenoiseSimple(dataset,k)

% Algorithm:
% 1- Run the plain denoising to obtain new set of points Dt'
% 2- Create the kNN graph G on Dt'
% 3- Run Isograph on G

distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

switch dataset
	case 'u'
		dsname = 'USPS', load 'dtset/USPS';
	case 'm'
		dsname = 'MNIST', load 'dtset/MNIST';
	case 'C'
		dsname = 'corel_cedd_10'; pc=1000; load 'dtset/corel_cedd_10';
	case 'a'
		dsname = 'alpha'; pc=1000; load 'dtset/alpha';	
	case 'g'
		dsname = 'g241c'; pc=1000; load 'dtset/g241c';	
	case 'G'
		dsname = 'g241n'; pc=1000; load 'dtset/g241n';	
	case 'd'
		dsname = 'digit1'; pc=1000; load 'dtset/digit1';
	case 'o'
		dsname = 'coil'; pc=1000; load 'dtset/coil';
	case 'O'
		dsname = 'coil2'; pc=1000; load 'dtset/coil2';
	case 'b'
		dsname = 'bci'; pc=1000; load 'dtset/bci';
end

N = size(Dt,1);  % number of instances
n = min(N,1000);		 % sample instance size
labels = labels + (1-min(labels)); % normalize the labels to 1,2,...
classN = max(labels);
cvFolds = round(n/(classN*10));
iterNum = cvFolds;	% number of iterations to run the algorithm
labeledNum = floor(n / cvFolds);

rndTotal = randperm(N);

DtSubset = Dt(rndTotal(1:n), :);
labelsSubset = labels(rndTotal(1:n), :);
pc = min(pc, size(DtSubset,2));

%% PCA
% disp('Principal Component Analysis ...');
DtSubset = DtSubset - repmat(mean(DtSubset), n, 1);
S = DtSubset' * DtSubset;
[E, V] = eig(S);
[~, Vsi] = sort(diag(V), 'descend');
E = E(:, Vsi(1:pc));
DtSubset = DtSubset * E;
%disp('-----------------------------------')

W = sparse(n,n);
WD = sparse(n,n);

W01 = sparse(n,n);
W01D = sparse(n,n);

for i = 1:n
	ithd = repmat(DtSubset(i,:), n, 1);
	if (distype == 1)
		dis = sqrt(sum((ithd-DtSubset).^2, 2));
	else
		dis = -sum(ithd.*DtSubset,2) ./ sqrt(sum(DtSubset.*DtSubset,2));
	end
	[disSorted, IX] = sort(dis);
	W(i, IX(2:k+1)) = disSorted(2:k+1);
	W(IX(2:k+1),i) = disSorted(2:k+1);
	W(i,i)=0;
		
	W01(i, IX(2:k+1)) = 1;
	W01(IX(2:k+1),i) = 1;
	W01(i,i) = 0;	
end

% Denoising
eps = .01;
err = eps+1;
maxIter = 3;
sigmaDenoising = 1;
kDenoising=5;
t=1;
kl=3;

while (err > eps) && (t < maxIter)
	%% computing Laplacian matrix
	WDenoising = sparse(n,n);
	for i = 1:n
		ithd = repmat(DtSubset(i,:), n, 1);
		if (distype == 1)
			dis = sqrt(sum((ithd-DtSubset).^2, 2));
		else
			dis = -sum(ithd.*DtSubset,2) ./ sqrt(sum(DtSubset.*DtSubset,2));
		end
		[~, IX] = sort(dis);
		WDenoising(i, IX(2:kDenoising+1)) = 1;
		WDenoising(IX(2:kDenoising+1), i) = 1;
		WDenoising(i, i) = 0;
	end
	D = sparse(diag(sum(WDenoising)));
	L = D \ (D - WDenoising);
	
	%% updating graph
	xLast = DtSubset;
	DtSubset = (eye(n) + sigmaDenoising*t*L)\DtSubset;
	err = norm(DtSubset - xLast);
	disp(['loop #' num2str(t) '; gradient = ' num2str(err) ';']);
	t = t+1;
end

for i = 1:n
	ithd = repmat(DtSubset(i,:), n, 1);
	if (distype == 1)
		dis = sqrt(sum((ithd-DtSubset).^2, 2));
	else
		dis = -sum(ithd.*DtSubset,2) ./ sqrt(sum(DtSubset.*DtSubset,2));
	end
	[disSorted, IX] = sort(dis);
	WD(i, IX(2:k+1)) = disSorted(2:k+1);
	WD(IX(2:k+1), i) = disSorted(2:k+1);
	WD(i,i)=0;
		
	W01D(i, IX(2:k+1)) = 1;
	W01D(IX(2:k+1), i) = 1;
	W01D(i,i) = 0;

	Wk(i, IX(2:kl+1)) = 1;
	Wk(IX(2:kl+1),i) = 1;
	Wk(i,i) = 0;	
end

WDI = IsographReweight(WD, Wk, WD > 0, n, 2.0, 3, 0.5, 2);

W01DI = W01D;
W01DI(abs(WDI-WD) > 0.1) = 0;

disp('Graph built');


acc01 = zeros(iterNum, 1);
accML = zeros(iterNum, 1);
accMLD = zeros(iterNum, 1);
acc01D = zeros(iterNum, 1);
acc01DI = zeros(iterNum, 1);
accMLDI = zeros(iterNum, 1);
	
L01 = (sparse(diag(sum(W01))) - W01);
L01D = (sparse(diag(sum(W01D))) - W01D);
L01DI = (sparse(diag(sum(W01DI))) - W01DI);

per = randperm(n);

for iter=1:iterNum
	rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
	if(iter==1)
		sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd), 1:labeledNum, 0.02)
		sigma1 = abs(sigma1);
	end
	
	[s f w] = find(W(rnd,rnd));
	WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
	LML = (sparse(diag(sum(WML))) - WML);

	if(iter==1)
		sigma2 = optimizeSigma(@ML, WD(rnd,rnd), labelsSubset(rnd), 1:labeledNum,0.02)
		sigma2 = abs(sigma2);
	end
  
	[s f w] = find(WD(rnd,rnd));
	WMLD = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
	LMLD = (sparse(diag(sum(WMLD))) - WMLD);
	
	if(iter==1)
		sigma3 = optimizeSigma(@ML, WDI(rnd,rnd), labelsSubset(rnd), 1:labeledNum,0.02)		
		sigma3 = abs(sigma3);
	end
  
	[s f w] = find(WDI(rnd,rnd));
	WMLDI = sparse(s, f, exp(-w.^2/(sigma3*sigma3)), n, n);
	LMLDI = (sparse(diag(sum(WMLDI))) - WMLDI);
	
	acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd));
	accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML);
	acc01D(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01D(rnd,rnd));
	accMLD(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLD);
	acc01DI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01DI(rnd,rnd));
	accMLDI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLDI);
end

fprintf('acc01 mean=%f stddev=%f\n', mean(acc01), sqrt(var(acc01)));
fprintf('accML mean=%f stddev=%f\n', mean(accML), sqrt(var(accML)));
fprintf('acc01D mean=%f stddev=%f\n', mean(acc01D), sqrt(var(acc01D)));
fprintf('accMLD mean=%f stddev=%f\n', mean(accMLD), sqrt(var(accMLD)));
fprintf('acc01DI mean=%f stddev=%f\n', mean(acc01DI), sqrt(var(acc01DI)));
fprintf('accMLDI mean=%f stddev=%f\n', mean(accMLDI), sqrt(var(accMLDI)));


acc01Mean = mean(acc01);
accMLMean = mean(accML);
acc01DMean = mean(acc01D);
accMLDMean = mean(accMLD);
acc01DIMean = mean(acc01DI);
accMLDIMean = mean(accMLDI);
