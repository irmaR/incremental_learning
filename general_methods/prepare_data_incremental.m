function [train,class,test_fea,binary_classes_test]=prepare_data_incremental(positive_class,train_fea,train_class,test_fea,test_class)
binary_classes_train = cell(length(train_class),1);
binary_classes_test = cell(length(test_class),1);

%just change labels into positive and negative
for i=1:length(train_class)
  if train_class(i)==positive_class
     binary_classes_train{i}=['pos'];
  else
     binary_classes_train{i}=['neg'];
  end
end

for i=1:length(test_class)
  if test_class(i)==positive_class
     binary_classes_test{i}=['pos'];
  else
     binary_classes_test{i}=['neg'];
  end
end

counter_pos=1;
counter_neg=1;
for i = 1:size(train_fea,1)      
    if binary_classes_train{i,1}=='pos'
        positive_selected(counter_pos,:)=train_fea(i,:);
        positive_class_sel(counter_pos,:)=binary_classes_train(i);
        counter_pos=counter_pos+1;
    else
        negative_selected(counter_neg,:)=train_fea(i,:);
        negative_class(counter_neg,:)=binary_classes_train(i);
        counter_neg=counter_neg+1;
    end
end

%oversample if #pos<#neg

if exist('positive_selected','var')
  rep=round(size(negative_selected,1)/length(positive_class_sel));
  %replicate selected result with positives #rep times
  positive_selected=reshape(repmat(positive_selected,1,rep),rep*size(positive_selected,1),size(positive_selected,2));
  positive_class_sel=reshape(repmat(positive_class_sel,1,rep),rep*size(positive_class_sel,1),size(positive_class_sel,2));
  train=cat(1,positive_selected,negative_selected);
  class=cat(1,positive_class_sel,negative_class);
else
  train=negative_selected;
  class=negative_class;
end

train=full(train);

fprintf('Returning')
end
