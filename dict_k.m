
function [re ind_f]=dict_k(x,k)
    [N m]=size(x);
    for i=1:N
        err=x(1:N,:)-repmat(x(i,:),N,1);
        err = err.*err;
        err = sum(err');
        b=err;
        [max ind]=sort(b);
        B=zeros(k,m);
        for j=2:k+1
            B(j-1,:)=x(ind(j),:);
        end
        B=B';
        ind_f(i,:)=ind;
        spb=x(i,:)';
        cvx_begin
		%s_quiet = cvx_quiet(true);
           variable xp(k);
%            minimize( norm( cof*xp, 1 ) );
           
           minimize(norm(B * xp - spb));
           subject to
           xp(:,1)>0;
        cvx_end        
        a2(i,:)=xp';
    end
    re=a2;
    re(re<10e-4)=0;
end


