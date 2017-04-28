function [accuracy,label,F1Score,score]=run_svm_experiment(train_fea,train_class,test_fea,test_class)
fprintf('Training the svm')
% 
%svm = svmtrain(train_fea,train_class,'kernel_function','rbf')
% label = svmclassify(svm,test_fea)
% shift = svm.ScaleData.shift;
% scale = svm.ScaleData.scaleFactor;
% x2 = bsxfun(@plus,test_fea,shift);
% x2 = bsxfun(@times,test_fea,scale);
% sv = svm.SupportVectors;
% alphaHat = svm.Alpha;
% bias = svm.Bias;
% kfun = svm.KernelFunction;
% kfunargs = svm.KernelFunctionArgs;
% 
% f = kfun(sv,x2,kfunargs{:})'*alphaHat(:) + bias;
% score = -f;

svm = fitcsvm(full(train_fea),full(train_class),'KernelFunction','rbf');

%cv = crossval(svm);
%[label,score] = kfoldPredict(cv);
[label,score] = predict(svm,full(test_fea));
%label = svmclassify(svm,test_fea);
label';
test_class';
true_positives=0;
true_negatives=0;
false_positives=0;
false_negatives=0;
sum1=0.0;
fprintf('\n')
for i=1:size(label,1)
    %fprintf('Predicted %d, True: %d\n',label(i),test_class(i)) 
    if label(i)==test_class(i)
      sum1=sum1+1.0;
    end
    if label(i)==1 && test_class(i)==1
        true_positives=true_positives+1;
    end
    if label(i)==1 && test_class(i)==-1
       false_positives=false_positives+1;
    end
    if label(i)==-1 && test_class(i)==-1
        true_negatives=true_negatives+1;
    end
    if label(i)==-1 && test_class(i)==1
        false_negatives=false_negatives+1;
    end
end
precision=true_positives/(true_positives+false_positives);
recall=true_positives/(true_positives+false_negatives);
accuracy=sum1/size(label,1);



if recall+precision>0
 F1Score=2*recall*precision/(recall+precision);
else
 F1Score=0;
end
fprintf('Macro F1:%d TP: %d, FP: %d, TN: %d, FN: %d, recall: %d, precision: %d,accuracy: %d',F1Score,true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy)

end