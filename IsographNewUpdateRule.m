function [acc01Mean accMLMean acc01IMean accMLIMean acc01NIMean accMLNIMean] = IsographNewUpdateRule(dataset,k)

distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

LoadDataset();

N = size(Dt,1);  % number of instances
n = min(N,2000);		 % sample instance size
classN = max(labels);
cvFolds = round(n/(classN*10));
iterNum = cvFolds;    % number of iterations to run the algorithm
labeledNum = floor(n / cvFolds);
labels = labels + (1-min(labels)); % normalize the labels to 1,2,...

Dt = double(full((Dt)));
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

kl=3;

W = sparse(n,n);
WI = sparse(n,n);
WNI = sparse(n,n);

W01 = sparse(n,n);
W01I = sparse(n,n);
W01NI = sparse(n,n);


Wn1= sparse(n,n);
Wk= sparse(n,n);

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

%update rule ghabl

W01I=W01;

WItemp = IsographReweight(W, Wk, W > 0, n, 2.0,3,0.5,1);
[s,f,~]=find(W > 0);

for i=1:length(s)
	WI(s(i),f(i))=WItemp(s(i),f(i));
	if (abs(WItemp(s(i),f(i))-W(s(i),f(i)))>0.1)
		W01I(s(i),f(i))=0;
	end
end

%update rule jadid
W01NI=W01;

WNItemp = IsographReweight(W, Wk, W > 0, n, 2.0,3,0.5,2);
[s,f,~]=find(W > 0);

for i=1:length(s)
	WNI(s(i),f(i))=WNItemp(s(i),f(i));
	if (abs(WNItemp(s(i),f(i))-W(s(i),f(i)))>0.1)
		W01NI(s(i),f(i))=0;
	end
end


disp('Graph built');


per = randperm(n);

acc01 = zeros(iterNum, 1);
accML = zeros(iterNum, 1);
accMLI = zeros(iterNum, 1);
acc01I = zeros(iterNum, 1);
acc01NI = zeros(iterNum, 1);
accMLNI = zeros(iterNum, 1);
    
dgnl = sum(W01);
De = sparse(diag(dgnl));
L01 = (De - W01);

dgnl = sum(W01I);
De = sparse(diag(dgnl));
L01I = (De - W01I);

dgnl = sum(W01NI);
De = sparse(diag(dgnl));
L01NI = (De - W01NI);

%LLLE01=(LLLE01+LLLE01')/2;
for iter=1:iterNum
    rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
	if(iter==1)
      sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd), 1:labeledNum, 0.02)
	  sigma1 = abs(sigma1);
	end
	
	[s f w] = find(W(rnd,rnd));
    WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
	dgnl = sum(WML);
    DML = sparse(diag(dgnl));
    LML = (DML - WML);

    if(iter==1)
       sigma2 = optimizeSigma(@ML, WI(rnd,rnd), labelsSubset(rnd), 1:labeledNum,0.02)
	   sigma2 = abs(sigma2);
    end
  
    [s f w] = find(WI(rnd,rnd));
    WMLI = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
    dgnl = sum(WMLI);
    DMLD = sparse(diag(dgnl));
    LMLI = (DMLD - WMLI);
    
	if(iter==1)
        sigma3 = optimizeSigma(@ML, WNI(rnd,rnd), labelsSubset(rnd), 1:labeledNum,0.02)        
		sigma3 = abs(sigma3);
	end
  
    [s f w] = find(WNI(rnd,rnd));
    WMLNI = sparse(s, f, exp(-w.^2/(sigma3*sigma3)), n, n);
    dgnl = sum(WMLNI);
    DMLDI = sparse(diag(dgnl));
    LMLNI = (DMLDI - WMLNI);
    
% 	disp(['sigma = ' num2str(sigma)])
                
    acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd));
    accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML);
    acc01I(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01I(rnd,rnd));
	accMLI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLI);
    acc01NI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01NI(rnd,rnd));
	accMLNI(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LMLNI);
end

fprintf('acc01 mean=%f stddev=%f\n', mean(acc01), sqrt(var(acc01)));
fprintf('accML mean=%f stddev=%f\n', mean(accML), sqrt(var(accML)));
fprintf('acc01n mean=%f stddev=%f\n', mean(acc01I), sqrt(var(acc01I)));
fprintf('accMLn mean=%f stddev=%f\n', mean(accMLI), sqrt(var(accMLI)));
fprintf('acc01nChanged mean=%f stddev=%f\n', mean(acc01NI), sqrt(var(acc01NI)));
fprintf('accMLnChanged mean=%f stddev=%f\n', mean(accMLNI), sqrt(var(accMLNI)));


acc01Mean = mean(acc01);
accMLMean = mean(accML);
acc01IMean = mean(acc01I);
accMLIMean = mean(accMLI);
acc01NIMean = mean(acc01NI);
accMLNIMean = mean(accMLNI);