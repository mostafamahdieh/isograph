function [acc01Mean1 acc01IMean1 acc01DMean1 acc01DIMean1 accMLMean1 accMLIMean1 accMLDMean1 accMLDIMean1 acc01Mean2 acc01IMean2 acc01DMean2 acc01DIMean2 accMLMean2 accMLIMean2 accMLDMean2 accMLDIMean2 pc n] = IsographAll(dataset,k)

distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions
regress = 0;

LoadDataset;

kl=3;
N = size(Dt,1);       % number of instances
n = min(N,1000);      % sample instance size
if (regress == 0)
	labels = labels + (1-min(labels)); % normalize the labels to 1,2,...
end

% if (regress == 1)
% 	cvFolds = 5;
% else
% 	classN = max(labels);
% 	cvFolds = round(n/(classN*10));
% end
% 	
% iterNum = cvFolds;	% number of iterations to run the algorithm
% labeledNum = floor(n / cvFolds);

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
WDI = sparse(n,n);

W01 = sparse(n,n);
W01D = sparse(n,n);
W01DI = sparse(n,n);
Wk = sparse(n,n);

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
		
	Wk(i, IX(2:kl+1)) = 1;
	Wk(IX(2:kl+1),i) = 1;
	Wk(i,i)=0;
end

WI = IsographReweight(W, Wk, W > 0, n, 2.0, 3, 0.5, 2);

W01I = W01;
W01I(abs(WI-W) > 0.1) = 0;

% Denoising
eps = .01;
err = eps+1;
maxIter = 3;
sigmaDenoising = 1;
kDenoising=k;
t=1;
DtSubsetCopy = DtSubset;
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
end

t=1;
DtSubset = DtSubsetCopy;
kDenoising = k;
err = eps+1;
while (err > eps) && (t < maxIter)
	%% computing Laplacian matrix
	WDenoisingW = sparse(n,n);
	WDenoising  = sparse(n,n);
	Wk  = sparse(n,n);
	for i = 1:n
		ithd = repmat(DtSubset(i,:), n, 1);
		if (distype == 1)
			dis = sqrt(sum((ithd-DtSubset).^2, 2));
		else
			dis = -sum(ithd.*DtSubset,2) ./ sqrt(sum(DtSubset.*DtSubset,2));
		end
		[disSorted, IX] = sort(dis);
		WDenoisingW(i, IX(2:kDenoising+1)) = disSorted(2:kDenoising+1);
		WDenoisingW(IX(2:kDenoising+1), i) = disSorted(2:kDenoising+1);
		WDenoisingW(i, i) = 0;
		
		WDenoising(i, IX(1:kDenoising+1)) = 1;
		WDenoising(IX(1:kDenoising+1), i) = 1;
		WDenoising(i, i) = 0;
		
		Wk(i, IX(2:kl+1)) = 1;
		Wk(IX(2:kl+1),i) = 1;
		Wk(i,i)=0;
	end
	
	WDenoisingI = IsographReweight(WDenoisingW, Wk, WDenoisingW > 0, n, 2.0, 3, 0.5, 2);
	[s,f,~]=find(WDenoisingW > 0);

	for i=1:length(s)
		if (abs(WDenoisingI(s(i),f(i))-WDenoisingW(s(i),f(i)))>0.1)
			WDenoising(s(i),f(i))=0;
		end
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
	WDI(i, IX(2:k+1)) = disSorted(2:k+1);
	WDI(IX(2:k+1),i) = disSorted(2:k+1);
	WDI(i,i)=0;
		
	W01DI(i, IX(2:k+1)) = 1;
	W01DI(IX(2:k+1),i) = 1;
	W01DI(i,i) = 0;	
end

disp('Graph built');


disp('Settin 1');


if (regress == 1)
	cvFolds = 5;
else
	classN = max(labels);
	cvFolds = round(n/(classN*20));
end
	
iterNum = cvFolds;	% number of iterations to run the algorithm
labeledNum = floor(n / cvFolds);





acc01 = zeros(iterNum, 1);
accML = zeros(iterNum, 1);
accMLI = zeros(iterNum, 1);
acc01I = zeros(iterNum, 1);
accMLD = zeros(iterNum, 1);
acc01D = zeros(iterNum, 1);
acc01DI = zeros(iterNum, 1);
accMLDI = zeros(iterNum, 1);
	
L01 = (sparse(diag(sum(W01))) - W01);
L01I = (sparse(diag(sum(W01I))) - W01I);
L01D = (sparse(diag(sum(W01D))) - W01D);
L01DI = (sparse(diag(sum(W01DI))) - W01DI);

per = randperm(n);

for iter=1:iterNum
	rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
	if(iter==1)
		sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, regress)
% 		sigma1=1000;
		sigma1 = abs(sigma1);
	end
	
	[s f w] = find(W(rnd,rnd));
	WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
	LML = (sparse(diag(sum(WML))) - WML);

	if(iter==1)
		sigma2 = optimizeSigma(@ML, WD(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum,0.02, regress)
% 		sigma2=1000;
		sigma2 = abs(sigma2);
	end
  
	[s f w] = find(WD(rnd,rnd));
	WMLD = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
	LMLD = (sparse(diag(sum(WMLD))) - WMLD);
	
	if(iter==1)
		sigma3 = optimizeSigma(@ML, WDI(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, regress)		
% 		sigma3=1000;
		sigma3 = abs(sigma3);
	end
  
	[s f w] = find(WDI(rnd,rnd));
	WMLDI = sparse(s, f, exp(-w.^2/(sigma3*sigma3)), n, n);
	LMLDI = (sparse(diag(sum(WMLDI))) - WMLDI);
	
	if(iter==1)
		sigma4 = optimizeSigma(@ML, WI(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, regress)		
% 		sigma4=1000;
		sigma4 = abs(sigma4);
	end
  
	[s f w] = find(WI(rnd,rnd));
	WMLI = sparse(s, f, exp(-w.^2/(sigma4*sigma4)), n, n);
	LMLI = (sparse(diag(sum(WMLI))) - WMLI);
	
	acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd), regress);
	accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML, regress);
	acc01I(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01I(rnd,rnd), regress);
	accMLI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLI, regress);
	acc01D(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01D(rnd,rnd), regress);
	accMLD(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLD, regress);
	acc01DI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01DI(rnd,rnd), regress);
	accMLDI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLDI, regress);
end

fprintf('acc011 mean=%f stddev=%f\n', mean(acc01), sqrt(var(acc01)));
fprintf('accML1 mean=%f stddev=%f\n', mean(accML), sqrt(var(accML)));
fprintf('acc01I1 mean=%f stddev=%f\n', mean(acc01I), sqrt(var(acc01I)));
fprintf('accMLI1 mean=%f stddev=%f\n', mean(accMLI), sqrt(var(accMLI)));
fprintf('acc01D1 mean=%f stddev=%f\n', mean(acc01D), sqrt(var(acc01D)));
fprintf('accMLD1 mean=%f stddev=%f\n', mean(accMLD), sqrt(var(accMLD)));
fprintf('acc01DI1 mean=%f stddev=%f\n', mean(acc01DI), sqrt(var(acc01DI)));
fprintf('accMLDI1 mean=%f stddev=%f\n', mean(accMLDI), sqrt(var(accMLDI)));


acc01Mean1 = mean(acc01);
accMLMean1 = mean(accML);
acc01IMean1 = mean(acc01I);
accMLIMean1 = mean(accMLI);
acc01DMean1 = mean(acc01D);
accMLDMean1 = mean(accMLD);
acc01DIMean1 = mean(acc01DI);
accMLDIMean1 = mean(accMLDI);



disp('Settin 2');

if (regress == 1)
	cvFolds = 5;
else
	classN = max(labels);
	cvFolds = round(n/(classN*10));
end
	
iterNum = cvFolds;	% number of iterations to run the algorithm
labeledNum = floor(n / cvFolds);






acc01 = zeros(iterNum, 1);
accML = zeros(iterNum, 1);
accMLI = zeros(iterNum, 1);
acc01I = zeros(iterNum, 1);
accMLD = zeros(iterNum, 1);
acc01D = zeros(iterNum, 1);
acc01DI = zeros(iterNum, 1);
accMLDI = zeros(iterNum, 1);
	
L01 = (sparse(diag(sum(W01))) - W01);
L01I = (sparse(diag(sum(W01I))) - W01I);
L01D = (sparse(diag(sum(W01D))) - W01D);
L01DI = (sparse(diag(sum(W01DI))) - W01DI);

per = randperm(n);

for iter=1:iterNum
	rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
	if(iter==1)
		sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, regress)
% 		sigma1=1000;
		sigma1 = abs(sigma1);
	end
	
	[s f w] = find(W(rnd,rnd));
	WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
	LML = (sparse(diag(sum(WML))) - WML);

	if(iter==1)
		sigma2 = optimizeSigma(@ML, WD(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum,0.02, regress)
% 		sigma2=1000;
		sigma2 = abs(sigma2);
	end
  
	[s f w] = find(WD(rnd,rnd));
	WMLD = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
	LMLD = (sparse(diag(sum(WMLD))) - WMLD);
	
	if(iter==1)
		sigma3 = optimizeSigma(@ML, WDI(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, regress)		
% 		sigma3=1000;
		sigma3 = abs(sigma3);
	end
  
	[s f w] = find(WDI(rnd,rnd));
	WMLDI = sparse(s, f, exp(-w.^2/(sigma3*sigma3)), n, n);
	LMLDI = (sparse(diag(sum(WMLDI))) - WMLDI);
	
	if(iter==1)
		sigma4 = optimizeSigma(@ML, WI(rnd,rnd), labelsSubset(rnd, :), 1:labeledNum, 0.02, regress)		
% 		sigma4=1000;
		sigma4 = abs(sigma4);
	end
  
	[s f w] = find(WI(rnd,rnd));
	WMLI = sparse(s, f, exp(-w.^2/(sigma4*sigma4)), n, n);
	LMLI = (sparse(diag(sum(WMLI))) - WMLI);
	
	acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd), regress);
	accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML, regress);
	acc01I(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01I(rnd,rnd), regress);
	accMLI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLI, regress);
	acc01D(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01D(rnd,rnd), regress);
	accMLD(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLD, regress);
	acc01DI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01DI(rnd,rnd), regress);
	accMLDI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLDI, regress);
end

fprintf('acc012 mean=%f stddev=%f\n', mean(acc01), sqrt(var(acc01)));
fprintf('accML2 mean=%f stddev=%f\n', mean(accML), sqrt(var(accML)));
fprintf('acc01I2 mean=%f stddev=%f\n', mean(acc01I), sqrt(var(acc01I)));
fprintf('accMLI2 mean=%f stddev=%f\n', mean(accMLI), sqrt(var(accMLI)));
fprintf('acc01D2 mean=%f stddev=%f\n', mean(acc01D), sqrt(var(acc01D)));
fprintf('accMLD2 mean=%f stddev=%f\n', mean(accMLD), sqrt(var(accMLD)));
fprintf('acc01DI2 mean=%f stddev=%f\n', mean(acc01DI), sqrt(var(acc01DI)));
fprintf('accMLDI2 mean=%f stddev=%f\n', mean(accMLDI), sqrt(var(accMLDI)));


acc01Mean2 = mean(acc01);
accMLMean2 = mean(accML);
acc01IMean2 = mean(acc01I);
accMLIMean2 = mean(accMLI);
acc01DMean2 = mean(acc01D);
accMLDMean2 = mean(accMLD);
acc01DIMean2 = mean(acc01DI);
accMLDIMean2 = mean(accMLDI);

