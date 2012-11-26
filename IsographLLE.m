function [acc01Mean accMLMean accLLE01Mean accLLEMLMean] = CVTikhonovRankedML

dataset='u';
k=5;
switch dataset
		case 'a'
			dsname = 'alphanumeric', load 'dtset/alphanumeric'; sigma=5.5;
		case 'f'
			dsname = 'ForestCover', load 'dtset/ForestCover';
		case 'g'
			dsname = 'Segmentation', load 'dtset/Segmentation';
		case 'e'
			dsname = 'Gender', load 'dtset/Gender';
		case 'G'
			dsname = 'Gen', load 'dtset/Gen';
		case 'y'
			dsname = 'HyperSpectral', load 'dtset/HyperSpectral';
		case 'H'
			dsname = 'HSpectral', load 'dtset/HSpectral';
		case 'i'
			dsname = 'ISOLET', load 'dtset/ISOLET'; sigma=5.5;
		case 'm'
			dsname = 'MNIST', load 'dtset/MNIST'; sigma=1100;
		case 'M'
			dsname = 'MSpec', load 'dtset/MSpec';
		case 'n'
			dsname = 'News', pc=100, load 'dtset/News';
		case 'N'
			dsname = 'NewsN', distype=2, pc=100, kVals=(30:5:60)', load 'dtset/NewsN';
		case 's'
			dsname = 'SAT', kVals=(5:10)', load 'dtset/SAT';
		case 'c'
			dsname = 'corel_ychen_10', pc=1000, load 'dtset/corel_ychen_10';
		case 'C'
			dsname = 'corel_cedd_10'; pc=1000; load 'dtset/corel_cedd_10';sigma= 5;			
		case 't'
			dsname = 'twomoon', load 'dtset/twomoon';
		case 'u'
			dsname = 'USPS', load 'dtset/USPS'; sigma=800;
		case 'w'
			dsname = 'WaveForm', load 'dtset/WaveForm';
		case 'W'
			dsname = 'WaveFormN', load 'dtset/WaveFormN';
		case 'R'
			dsname = 'Wfrobust', load 'dtset/Wfrobust';
        case 'p'
			dsname = 'Pose_H', load 'dtset/Pose_H';
%     case 'b'
%             dsname = 'banana1_b', load 'mat/banana1_b'; Dt=data; labels=label_b;
	end

distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions
k2 = k;

N = size(Dt,1);  % number of instances
n = 1000;		 % sample instance size
classN = max(labels);
cvFolds = round(n/(classN*10));
iterNum = cvFolds;    % number of iterations to run the algorithm
%iterNum=1;
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
%disp('done');
%disp('-----------------------------------')

W = sparse(n,n);
W01 = sparse(n,n);

%kl =3;
 WLLE = sparse(n,n);
  WLLE01 = sparse(n,n);
[W_ ind]=dict_k(DtSubset,k);
for i=1:n
	for j=1:k
		if (W_(i,j) > 0)
			WLLE(i,ind(i,j+1))=W_(i,j);
			WLLE(ind(i,j+1),i)=W_(i,j);
			WLLE01(i,ind(i,j+1))=1;
			WLLE01(ind(i,j+1),i)=1;
		end
	end
end

WLLE=(WLLE+WLLE')/2;
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

disp('Graph built');


per = randperm(n);
% WnChanged = MSTReweightSimple(Wfull2, W01, n, 2.0,3);
% [s,f,w]=find(W2 > 0);
% 
% for i=1:length(s)
%     Wn(s(i),f(i))=WnChanged(s(i),f(i));
% end

accML = zeros(iterNum, 1);
acc01 = zeros(iterNum, 1);
accLLEML = zeros(iterNum, 1);
accLLE01 = zeros(iterNum, 1);
    
dgnl = sum(W01);
De = sparse(diag(dgnl));
L01 = (De - W01);


dgnl = sum(WLLE01);
De = sparse(diag(dgnl));
LLLE01 = (De - WLLE01);

%LLLE01=(LLLE01+LLLE01')/2;
for iter=1:iterNum
    rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
	if(iter==1)
      sigma1 = optimizeSigma(@ML, W(rnd,rnd), labelsSubset(rnd), 1:labeledNum, 0.02)
	  %sigma1=800;
    end
    [s f w] = find(W(rnd,rnd));
    WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
    dgnl = sum(WML);
    DML = sparse(diag(dgnl));
    LML = (DML - WML);


    if(iter==1)
        sigma2 = optimizeSigma(@ML, WLLE(rnd,rnd), labelsSubset(rnd), 1:labeledNum,0.02)        
	   %sigma2=800;
    end
  
    [s f w] = find(WLLE(rnd,rnd));
    WLLEML = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
    dgnl = sum(WLLEML);
    DMLN = sparse(diag(dgnl));
    LLLEML = (DMLN - WLLEML);
    
    
% 	disp(['sigma = ' num2str(sigma)])
                
    accML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LML);
    acc01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd));
	accLLEML(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LLLEML);
    accLLE01(iter) = laplacianLS(labelsSubset(rnd, :), labeledNum, LLLE01(rnd,rnd));
% 	fprintf('acc=%f accML=%f accMLN=%f\n', acc(iter), accML(iter), accMLN(iter));
end

fprintf('acc01 mean=%f stddev=%f\n', mean(acc01), sqrt(var(acc01)));
fprintf('accML mean=%f stddev=%f\n', mean(accML), sqrt(var(accML)));
fprintf('accLLE01 mean=%f stddev=%f\n', mean(accLLE01), sqrt(var(accLLE01)));
fprintf('accLLEML mean=%f stddev=%f\n', mean(accLLEML), sqrt(var(accLLEML)));


acc01Mean = mean(acc01);
accMLMean=mean(accML);
accLLE01Mean = mean(accLLE01);
accLLEMLMean = mean(accLLEML);