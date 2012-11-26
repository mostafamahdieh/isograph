runNum = 10;
acc01  = zeros(runNum, 1);
acc01I = zeros(runNum, 1);
accML  = zeros(runNum, 1);
accMLI = zeros(runNum, 1);

n = 1000;
dataset = 'u';

switch dataset
	case 'm'
		dsname = 'MNIST';
		load 'dtset/MNIST';
	case 'u'
		dsname = 'USPS';
		load 'dtset/USPS';
end

RandStream.setDefaultStream(RandStream('mt19937ar','seed',1390));
filename = sprintf('%s_subset_%d.txt', dsname, n);
handle = fopen(filename,'w');
if (handle < 0)
	fprintf('error on %s\r\n', ds);
	return;
end
fprintf(handle, '%d %d\r\n', runNum, n);
N = size(Dt,1);
for i=1:runNum
	rndTotal = randperm(N);
	for j=1:n
		fprintf(handle, '%d ', rndTotal(j));
	end
	fprintf(handle, '\r\n');
	randperm(n);
end
fclose(handle);