function W1 = IsographReweightNewRule(W,Wk,WE, n,beta,iterNum,alpha)

for it=1:iterNum
     numChecked=0;
     numUpdated=0;
     
    [s f w] = find(W);
    [w,ind]=sort(w,'ascend'); % edges have changed so we must sort them again
    s=s(ind);
    f=f(ind);
    m = size(s,1);	
	
	if (it == 1)
		disp(['number of edges ' num2str(m)]);
	end
	
	% find the minimum spanning tree of G
    [Tree, ~] = graphminspantree(W);
    delta=max(Tree(Tree~=0))*0.5;
    delta=delta*alpha;
	
    W1 = sparse(n,n);
	lastW = -1;
      
    for k=2:m
		if (w(k-1) ~= lastW) % add bigger edges to the subgraph
			W1(W == w(k-1)) = w(k-1);
			lastW = w(k-1);
		end
		
		if (WE(s(k),f(k)) == 0) % if this edge is an edge added on the graph-completion phase
								% dont change them
			continue;
		end
		
        [dist, ~, ~] = graphshortestpath(W1, s(k), f(k), 'Method','BFS');
        flen=dist;

        if (flen>2  && Wk(s(k),f(k))==0)
            
            if (w(k) >= 2*delta)
				if(flen>100)
					flen=100;
				end
                w(k) = max((w(k)-delta)*beta,(flen-1)*(w(k)-2*delta)+2*delta);
                W1(s(k),f(k))=w(k); % update the subgraph online
                numUpdated=numUpdated+1;		
            end
            numChecked = numChecked + 1;
        end
    end
    [it,numChecked,numUpdated]
   	W = W1;
	
	if(numUpdated==0)
		break;
	end
end