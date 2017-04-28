function F1_score_micro=get_F1_micro_score(predictions,test_class,labels)
%create matrix with binary decisions for test class
fprintf('TEST CLASS')
%test_class
fprintf('labels:')
%labels
for j=1:size(predictions,1)
for i=1:length(test_class) 
    if test_class(i)==labels(j)
            bin_test(j,i)=1;
    else
            bin_test(j,i)=0;
    end
end
end
TP=0;
FN=0;
TN=0;
FP=0;
for i=1:size(bin_test,1)
  for j=1:size(bin_test,2)
    %fprintf('Prediction %d, ground truth:%d\n',predictions(i,j),bin_test(i,j))
    if bin_test(i,j)==1 && predictions(i,j)==1
        TP=TP+1;
    elseif bin_test(i,j)==0 && predictions(i,j)==1
        FP=FP+1;
    elseif bin_test(i,j)==0 && predictions(i,j)==-1
        TN=TN+1;
    elseif bin_test(i,j)==1 && predictions(i,j)==-1
        FN=FN+1;
    end
  end
end
%fprintf('Micro score')
precision=TP/(TP+FP);
recall=TP/(TP+FN);
fprintf('Micro F1: TP: %d, FP: %d, TN: %d, FN: %d, recall: %d, precision: %d',TP,FP,TN,FN,recall,precision)

if recall+precision>0
  F1_score_micro=2*recall*precision/(recall+precision)
else
  F1_score_micro=0
end
end