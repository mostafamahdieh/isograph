dataset='u';
k=10;

distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

LoadDataset();

cond1=0;
cond2=0;
cond3=0;

RandStream.setDefaultStream(RandStream('mt19937ar','seed',1390));
fileId = fopen('Betweenness1.txt','a');
fprintf(fileId,'%s %d\r\n',dataset,k);


N = size(Dt,1);  % number of instances
n = min(N,1000);	 % sample instance size
labels = labels + (1-min(labels)); % normalize the labels to 1,2,...
classN = max(labels);
cvFolds = round(n/(classN*50));
iterNum = cvFolds;	% number of iterations to run the algorithm
labeledNum = floor(n / cvFolds);

rndTotal = randperm(N);

DtSubset = Dt(rndTotal(1:n), :);
labelsSubset = labels(rndTotal(1:n), :);

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


per = randperm(n);

DtSubsetNorm = sqrt(sum(DtSubset .^ 2,2));

for i = 1:n
	ithd = repmat(DtSubset(i,:), n, 1);
	ithdNorm = DtSubsetNorm(i);
	if (distype == 1)
		dis = sqrt(sum((ithd-DtSubset).^2, 2));
	else
		dis = 1.001-sum(ithd.*DtSubset,2) ./ (DtSubsetNorm * ithdNorm);
	end
	[disSorted, IX] = sort(dis);
	W(i, IX(2:k+1)) = disSorted(2:k+1);
	W(IX(2:k+1),i) = disSorted(2:k+1);
	W(i,i)=0;
		
	W01(i, IX(2:k+1)) = 1;
	W01(IX(2:k+1),i) = 1;
	W01(i,i) = 0;	

end
flag2=true;
while(flag2)
	d=zeros(n,n); %tule kutahtarin masir
	r=zeros(n,n); %tedade kutahtarin masir
	connected=zeros(n,n); %tedade kole masir

	between=zeros(n,n);
	
	cond=0;


	for i=1:n
		qu=zeros(n,1);
		squ=1;
		equ=1;
		visit=zeros(n,1);
		barresi=zeros(n,1);

		visit(i,1)=1;

		flag=true;
		visitnode=i;
		while(flag)
			barresi(visitnode)=1;
			ind=find(W01(visitnode,:)>0); %hamsayeha
			indtemp=find(barresi(ind)>0);
			ind1=ind(indtemp); %barresi shode
			ind2=setdiff(ind,ind1); %barresi nashode

			indtemp=find(visit(ind)>0);
			ind3=ind(indtemp); %visit shode
			ind4=setdiff(ind,ind3); %visit nashode
			%setdiff(ind4,ind2)
			if(visitnode~=i)
				ll=min(d(i,ind1));
				d(i,visitnode)=ll+1;

				indtemp=find(d(i,ind1)==ll);
				ind5=ind1(indtemp); %hamsaye dar kutahtarin masir

				r(i,visitnode)=sum(r(i,ind5));
				%r(i,visitnode)=sum(r(i,ind3));
			else
				d(i,visitnode)=0;
				r(i,visitnode)=1;
			end

			for j=ind4
				visit(j)=1;
				qu(equ)=j;
				equ=equ+1;
			end

			for j=ind3
				if(j~=visitnode)
					connected(i,j)=1;
				end
			end

			if(squ<equ)
				visitnode=qu(squ);
				squ=squ+1;
			else
				flag=false;
			end
		end
		
		cond=cond+max(d(i,:));

	end


	disp('betweenness');
	[ind1,ind2]=find(W01>0);

	for ll=1:length(ind1)
		x=ind1(ll);
		y=ind2(ll);
		for u=1:n
			for v=1:n
				if(u~=v && r(u,v)~=0)
					if(d(u,v)==d(u,x)+1+d(v,y))
						if( u==x && v==y)
							rtemp=1;
						else
							if(u==x)
								rtemp=r(y,v);
							else
								if(v==y)
									rtemp=r(u,x);
								else
									rtemp=r(u,x)*r(y,v);
								end
							end
						end
					between(x,y)=between(x,y)+ rtemp/r(u,v);
					end
				end
			end
		end
	end


	betweenness1=between.*connected;
	mm=max(max(betweenness1));

	if(mm==0)	
		flag2=false;
	else
	[l1,l2]=find(betweenness1==mm);
	W01(l1,l2)=0;
	W01(l2,l1)=0;
	
	[l1,l2]
	fprintf(fileId,'%d %d\r\n',l1,l2);
	cond
	fprintf(fileId,'%d\r\n',cond);
	cond1=cond2;
	cond2=cond3;
	cond3=cond;
	
	if(cond3<cond1)
		flag2=false;
	end

	end
	
end

% acc01 = zeros(iterNum, 1);
% L01 = (sparse(diag(sum(W01))) - W01);
% rnd = circshift(per,0);
% acc01 = laplacianLS(labelsSubset(rnd, :), labeledNum, L01(rnd,rnd))
