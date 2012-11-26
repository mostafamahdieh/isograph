runNum = 10;
acc01  = zeros(runNum, 1);
accML  = zeros(runNum, 1);

fileId = fopen('gnn.txt', 'a');
labeledPerClass = 10;
for ds =['u'];
	first = 1;
    for k=9:9
        RandStream.setDefaultStream(RandStream('mt19937ar','seed',1390));
		for i=1:runNum
           [k, i]
           [acc01(i), accML(i), pc, n] = GNNTikhonov(ds,k,i,labeledPerClass);
		end
		if (first == 1)
			fprintf(fileId, '%s n=%d pc=%d labeledPerClass=%d\r\n', ds, n, pc, labeledPerClass);
			first = 0;
		end
        fprintf(fileId, '%s %d %f %f\r\n', ds, k, mean(acc01), mean(accML));
    end
end