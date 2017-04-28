function [ Scores,macro_F_scores,all_predictions,a,F1_score_micro ] = run_svm_classifier(training_data,training_class,test_data,test_class,labels)
%RUN_SVM_CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
for i=1:length(labels)
         pos_class=labels(i);
         %[training_fea,training_classes,test_fea,test_classes]=prepare_data(pos_class,smpRank,train_fea,train_class,test_fea,test_class);
         [training_fea,training_classes,test_fea,test_classes]=prepare_data_svm(pos_class,full(training_data),full(training_class),test_data,test_class);
         %[accuracy,predictions,F1_macro,pihat_x,sp_x,labels_x]=run_experiment_instance(pos_class,samples(j),training_fea,training_classes,test_fea,test_classes);
         [accuracy,predictions,F1_macro,score]=run_svm_experiment(training_fea,training_classes,test_fea,test_classes);
         Scores(:,i) = score(:,2);
         macro_F_scores(i)=F1_macro;
         all_predictions(i,:)=predictions;    
end
[~,maxScore] = max(Scores,[],2);
for i=1:length(maxScore)
     maxScore(i)=labels(maxScore(i));
end
a=sum(~(full(test_class)~=maxScore))/size(maxScore,1);
fprintf('\nACCURACY:%f\n',a);
F1_score_micro=get_F1_micro_score(all_predictions,test_class,labels);
end

