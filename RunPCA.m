distype = 1; % 1 is for euclidian distance and 2 is for tf-id distance
pc = 35; % number of pca dimensions

dataset = 'u';

switch dataset
	case 'm'
		dsname = 'MNIST', load 'dtset/MNIST';
	case 'u'
		dsname = 'USPS', load 'dtset/USPS';
end

N = size(Dt,1);  % number of instances
Dt = Dt - repmat(mean(Dt), N, 1);
S = Dt' * Dt;
[E, V] = eig(S);
[~, Vsi] = sort(diag(V), 'descend');
E = E(:, Vsi(1:pc));
Dt = Dt * E;
filename = sprintf('%s_PCA_Dt_%d.txt', dsname, pc);
fileId = fopen(filename, 'w');
fprintf(fileId, '%d %d\r\n', N, pc);
for i=1:N
	for j=1:pc
		fprintf(fileId, '%d ', Dt(i,j));
	end
	fprintf(fileId, '\r\n');
end
fclose(fileId);