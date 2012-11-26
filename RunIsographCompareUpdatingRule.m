

runNum = 10;
acc = zeros(runNum, 1);
acc01D=zeros(runNum, 1);
acc01DI=zeros(runNum, 1);
accML = zeros(runNum, 1);
accMLD = zeros(runNum, 1);
accMLDI= zeros(runNum, 1);

fileId = fopen('resultNewRule-1000-NEWML.txt','a');
%for ds =['u'];
for ds =[ 'm' 'u' 'C' 'a' 'u' 'm'];
	range = 5:2:17
    for k=range
        RandStream.setDefaultStream(RandStream('mt19937ar','seed',1390));
        for i=1:runNum
            [k,i]
           [acc(i), accML(i), acc01I(i) accMLI(i) acc01NI(i) accMLNI(i)] = IsographNewUpdateRule(ds,k);
           %fprintf('acc=%f accN=%f\n', acc(i), accN(i));
        end
        fprintf(fileId, '%s %d %f %f %f %f %f %f\r\n', ds, k,mean(acc), mean(acc01I), mean(acc01NI), mean(accML), mean(accMLI), mean(accMLNI));
		%fprintf(fileId, '%s %d %f %f %f %f\r\n', ds, k,var(acc01D-acc), var(acc01D), var(accMLD-accML), var(accMLD));
    end
end