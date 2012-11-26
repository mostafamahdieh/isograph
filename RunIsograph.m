runNum = 10;
acc01  = zeros(runNum, 1);
acc01I = zeros(runNum, 1);
accML  = zeros(runNum, 1);
accMLI = zeros(runNum, 1);

fileId = fopen('result_gen.txt','a');
%for ds =['u'];
for ds =['g'];
    for k=7:2:17
        RandStream.setDefaultStream(RandStream('mt19937ar','seed',1390));
        for i=1:runNum
           [k, i]
           [acc01(i), accML(i), acc01I(i), accMLI(i)] = IsographNewUpdateRule(ds,k);
        end
        fprintf(fileId, '%s %d %f %f %f %f\r\n', ds(1), k, mean(acc01), mean(accML), mean(acc01I), mean(accMLI));
    end
end