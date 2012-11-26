

runNum = 10;
acc01 = zeros(runNum, 1);
acc01D=zeros(runNum, 1);
acc01DI=zeros(runNum, 1);
accML = zeros(runNum, 1);
accMLD = zeros(runNum, 1);
accMLDI= zeros(runNum, 1);

fileId = fopen('resultDenoising2000-new1.txt','a');
for ds =['O'];
    for k=7:2:17
        RandStream.setDefaultStream(RandStream('mt19937ar','seed',1390));
        for i=1:runNum
           [k, i]
           [acc01(i) acc01D(i) acc01DI(i)  accML(i) accMLD(i) accMLDI(i)] = IsographDenoise(ds,k);
        end
        fprintf(fileId, '%s %d %f %f %f %f %f %f\r\n', ds, k,mean(acc01),  mean(acc01D) , mean(acc01DI), mean(accML), mean(accMLD), mean(accMLDI));
    end
end