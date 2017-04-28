function [prediction,accuracy,precision,recall,F1score]=get_accuracy(predictions,ground_truth,labels)
probs=[]
for i=1 : size(predictions,1)
    max_index=find(predictions(i,:)==max(predictions(i,:)));
    %What to do when all prediction either 0 or NaN????
    if size(max_index,2)>1
        e=randperm(size(max_index,2));
        max_index=max_index(e(1));
    end
    prediction(i)=labels(max_index);
    probs(i)=predictions(i,max_index);
end
sum1=0.0
true_positives=0;
true_negatives=0;
false_positives=0;
false_negatives=0;

for i=1:size(prediction,2)
    %fprintf('Prediction %s, ground truth:%s\n',prediction{1,i},ground_truth{i})
    if prediction{1,i}==ground_truth{i}
      sum1=sum1+1.0;
    end

    if strcmp(prediction{1,i},'pos') && strcmp(ground_truth{i},'pos')
        true_positives=true_positives+1;
    end
    if strcmp(prediction{1,i},'pos') && strcmp(ground_truth{i},'neg')
       false_positives=false_positives+1;
    end
    if strcmp(prediction{1,i},'neg') && strcmp(ground_truth{i},'neg')
        true_negatives=true_negatives+1;
    end
    if strcmp(prediction{1,i},'neg') && strcmp(ground_truth{i},'pos')
        false_negatives=false_negatives+1;
    end
end
precision=true_positives/(true_positives+false_positives);
recall=true_positives/(true_positives+false_negatives);
accuracy=sum1/size(prediction,2);
fprintf('Macro F1: TP: %d, FP: %d, TN: %d, FN: %d, recall: %d, precision: %d,accuracy: %d',true_positives,false_positives,true_negatives,false_negatives,recall,precision,accuracy)

if recall+precision>0
 F1score=(2*recall*precision)/(recall+precision);
else
 F1score=0
end

       