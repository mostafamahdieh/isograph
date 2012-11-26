% Manifold Denoising

%% INIT
clc
load USPS
k = 5;
k1=5;
eps = .01;
err = eps+1;
n = 500;
maxIter = 2;
t = 1;
sigma = 1;
idx = randperm(n);
x = Dt(idx, :);
xOrg = x;
xLast = zeros(size(x));
y = labels(idx);

WML=sparse(n,n);
W01=sparse(n,n);
WMLn=sparse(n,n);
W01n=sparse(n,n);
%%before denoising
DtSubset=x;
distype = 1;
iterNum=1;
labeledNum=100;


for i = 1:n
	ithd = repmat(DtSubset(i,:), n, 1);
	if (distype == 1)
		dis = sqrt(sum((ithd-DtSubset).^2, 2));
	else
		dis = -sum(ithd.*DtSubset,2) ./ sqrt(sum(DtSubset.*DtSubset,2));
	end
	[disSorted, IX] = sort(dis);
	WML(i, IX(2:k+1)) = disSorted(2:k+1);
	WML(IX(2:k+1),i) = disSorted(2:k+1);
    WML(i,i)=0;
        
	W01(i, IX(2:k+1)) = 1;
	W01(IX(2:k+1),i) = 1;
    W01(i,i) = 0;
    
end


%% main
while (err > eps) && (t < maxIter)
    %% computing Laplacian matrix
    W = sparse(n,n);
    for i = 1:n
        ithd = repmat(x(i,:), n, 1);
        dis = sum((ithd-x).^2, 2);
        [~, IX] = sort(dis);
        W(i, IX(1:k1+1)) = 1;
        W(IX(1:k1+1), i) = 1;
        W(i, i) = 0;
    end
dgnl = sum(W);
D = sparse(diag(dgnl));
L = D \ (D - W);
    %% updating graph
    xLast = x;
    x = (eye(n) + sigma*t*L)\x;
    err = norm(x - xLast);
    disp(['loop #' num2str(t) '; gradient = ' num2str(err) ';']);
    t = t+1;
end

DtSubset=x;

for i = 1:n
	ithd = repmat(DtSubset(i,:), n, 1);
	if (distype == 1)
		dis = sqrt(sum((ithd-DtSubset).^2, 2));
	else
		dis = -sum(ithd.*DtSubset,2) ./ sqrt(sum(DtSubset.*DtSubset,2));
	end
	[disSorted, IX] = sort(dis);
	WMLn(i, IX(2:k+1)) = disSorted(2:k+1);
	WMLn(IX(2:k+1),i) = disSorted(2:k+1);
    WMLn(i,i)=0;
        
	W01n(i, IX(2:k+1)) = 1;
	W01n(IX(2:k+1),i) = 1;
    W01n(i,i) = 0;
    
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
accMLn = zeros(iterNum, 1);
acc01n = zeros(iterNum, 1);
    
dgnl = sum(W01);
De = sparse(diag(dgnl));
L01 = (De - W01);

iter=1;
dgnl = sum(W01n);
De = sparse(diag(dgnl));
L01n = (De - W01n);

 rnd = circshift(per,ones(n,1)*(iter-1)*labeledNum);
	if(iter==1)
      sigma1 = optimizeSigma(@ML, WML(rnd,rnd), y(rnd), 1:labeledNum, 0.02)
	  %sigma1=800;
    end
    [s f w] = find(WML(rnd,rnd));
    WML = sparse(s, f, exp(-w.^2/(sigma1*sigma1)), n, n);
    dgnl = sum(WML);
    DML = sparse(diag(dgnl));
    LML = (DML - WML);


    if(iter==1)
        sigma2 = optimizeSigma(@ML, WMLn(rnd,rnd), y(rnd), 1:labeledNum,0.02)        
	   %sigma2=800;
    end
  
    [s f w] = find(WMLn(rnd,rnd));
    WMLn = sparse(s, f, exp(-w.^2/(sigma2*sigma2)), n, n);
    dgnl = sum(WMLn);
    DMLN = sparse(diag(dgnl));
    LMLn = (DMLN - WMLn);
    
    
% 	disp(['sigma = ' num2str(sigma)])
                
    accML(iter) = laplacianLS(y(rnd, :), labeledNum, LML);
    acc01(iter) = laplacianLS(y(rnd, :), labeledNum, L01(rnd,rnd));
	accMLn(iter) = laplacianLS(y(rnd, :), labeledNum, LMLn);
    acc01n(iter) = laplacianLS(y(rnd, :), labeledNum, L01n(rnd,rnd));
% 	fprintf('acc=%f accML=%f accMLN=%f\n', acc(iter), accML(iter),
% 	accMLN(iter));

fprintf('acc01 mean=%f stddev=%f\n', mean(acc01), sqrt(var(acc01)));
fprintf('accML mean=%f stddev=%f\n', mean(accML), sqrt(var(accML)));
fprintf('acc01n mean=%f stddev=%f\n', mean(acc01n), sqrt(var(acc01n)));
fprintf('accMLn mean=%f stddev=%f\n', mean(accMLn), sqrt(var(accMLn)));
