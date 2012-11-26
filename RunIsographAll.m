runNum = 10;
acc011 = zeros(runNum, 1);
acc01I1=zeros(runNum, 1);
acc01D1=zeros(runNum, 1);
acc01DI1=zeros(runNum, 1);
accML1 = zeros(runNum, 1);
accMLI1 = zeros(runNum, 1);
accMLD1 = zeros(runNum, 1);
accMLDI1= zeros(runNum, 1);

acc012 = zeros(runNum, 1);
acc01I2=zeros(runNum, 1);
acc01D2=zeros(runNum, 1);
acc01DI2=zeros(runNum, 1);
accML2 = zeros(runNum, 1);
accMLI2 = zeros(runNum, 1);
accMLD2 = zeros(runNum, 1);
accMLDI2= zeros(runNum, 1);

fileId = fopen('result-TestDenoishing.txt', 'a');
for ds = ['u', 'm']
	first = 1;
    for k=7:2:15
	%for k=15
		seed = 1390;
        RandStream.setDefaultStream(RandStream('mt19937ar','seed',seed));
		for i=1:runNum
			[k, i]
           [acc011(i) acc01I1(i) acc01D1(i) acc01DI1(i)  accML1(i) accMLI1(i) accMLD1(i) accMLDI1(i) acc012(i) acc01I2(i) acc01D2(i) acc01DI2(i)  accML2(i) accMLI2(i) accMLD2(i) accMLDI2(i) pc n] = IsographAll(ds,k);
		end
		if (first == 1)
			fprintf(fileId, '%s %s n=%d pc=%d seed=%d (01 01I 01D 01DI ML MLI MLD MLDI)\r\n', datestr(now), ds, n, pc, seed);
			first = 0;
		end
        fprintf(fileId, '%s %d %f %f %f %f %f %f %f %f\r\n', ds, k, mean(acc011), mean(acc01I1), mean(acc01D1) , mean(acc01DI1), mean(accML1), mean(accMLI1), mean(accMLD1), mean(accMLDI1));
		fprintf(fileId, '%s %d %f %f %f %f %f %f %f %f\r\n', ds, k, mean(acc012), mean(acc01I2), mean(acc01D2) , mean(acc01DI2), mean(accML2), mean(accMLI2), mean(accMLD2), mean(accMLDI2));
    end
end