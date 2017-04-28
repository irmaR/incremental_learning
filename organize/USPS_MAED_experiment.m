%MAED experiment
clear all;
load('data/USPS.mat');
%number of samples to use

train_percentage=80;
test_percentage=20;

%split the data
ix = randperm(size(fea,1));
end_training=round(train_percentage*size(fea,1)/100);
training_ix=ix(1:end_training);
test_ix=ix(end_training:length(ix));

train_fea=fea(training_ix,:);
train_class=gnd(training_ix,:);
test_fea=fea(test_ix,:);
test_class=gnd(test_ix,:);

%samples to try
samples=[40,80,120,160];
labels=sort(unique(gnd));

F1_macro_scores=[];
F1_micro_scores=[];
results_per_sample=zeros(length(samples),length(labels)); % a matrix containing class related F1scores  
time_per_sample_size=[];

%go through each label
class_results=[];
for j=1:length(samples)
    macro_F_scores=[];
    tic
    smpRank=select_sample(train_fea,samples(j));
    time_per_sample_size(j)=toc
    
for i=1:length(labels)
     pos_class=labels(i);
     [training_fea,training_classes,test_fea,test_classes]=prepare_data(pos_class,smpRank,train_fea,train_class,test_fea,test_class);
     %[accuracy,predictions,F1_macro,pihat_x,sp_x,labels_x]=run_experiment_instance(pos_class,samples(j),training_fea,training_classes,test_fea,test_classes);
     [accuracy,predictions,F1_macro]=run_svm_experiment(training_fea,training_classes,test_fea,test_classes);
     macro_F_scores(i)=F1_macro;
     all_predictions(i,:)=predictions;    
end
   F1_score_micro=get_F1_micro_score(all_predictions,test_class,labels);
   F1_micro_scores(j)=F1_score_micro;
   F1_macro_scores(j)=mean(macro_F_scores);
   results_per_sample(j,:)=macro_F_scores;
end
F1_micro_scores
F1_macro_scores
    


