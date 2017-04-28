%MAED experiment
clear all;
clc;
load('data/RCV1_4Class.mat');
s = RandStream('mt19937ar','Seed',0);
rand('twister',5489);	
fprintf('Data loaded...')
%number of samples to use

train_percentage=80;
test_percentage=20;

%split the data
N=size(fea,1);
%for now only select K points
ix = randperm(s,N)';
K=4000;
N=K;
ix=ix(1:K,:);
end_training=round(train_percentage*N/100);
training_ix=ix(1:end_training);
test_ix=ix(end_training:length(ix));

train_fea=fea(training_ix,:);
train_class=gnd(training_ix,:);

%test_fea=train_fea;
%test_class=train_class;


test_fea=fea(test_ix,:);
test_class=gnd(test_ix,:);

%samples to try
samples=[100,200,300,400,500,600,700,800];
%samples=[size(train_fea,1)];
batches=20*ones(1,8);
%batches=[800,800,800,800,800];
labels=sort(unique(gnd));

F1_macro_scores=[];
F1_micro_scores=[];
results_per_sample=zeros(length(samples),length(labels)); % a matrix containing class related F1scores  
time_per_sample_size=[];

%go through each label
class_results=[];
accuracies=[];
for j=1:length(samples)
    macro_F_scores=[];
    tic
    fprintf('Selecting incrementally the most informative vectors...')
    fprintf('Using %d samples for kernel in each batch\n',samples(j))
    [train_fea_sub,train_class_sub]=select_sample_incremental(train_fea,train_class,samples(j),batches(j));
    %[train_fea_sub,train_class_sub,smpRank,values] = select_sample(train_fea,train_class,samples(j));
    
    %just take cut
    %train_fea_sub=train_fea(1:samples(j),:);
    %train_class_sub=train_class(1:samples(j),:);
    
    fprintf('Selected %d training examples, now running classifier',size(train_fea_sub,1))
    time_per_sample_size(j)=toc
    
for i=1:length(labels)
     pos_class=labels(i);
     %[training_fea,training_classes,test_fea,test_classes]=prepare_data(pos_class,smpRank,train_fea,train_class,test_fea,test_class);
     [training_fea,training_classes,test_fea,test_classes]=prepare_data_svm(pos_class,train_fea_sub,train_class_sub,test_fea,test_class);
     %[accuracy,predictions,F1_macro,pihat_x,sp_x,labels_x]=run_experiment_instance(pos_class,samples(j),training_fea,training_classes,test_fea,test_classes);
     [accuracy,predictions,F1_macro,score]=run_svm_experiment(training_fea,training_classes,test_fea,test_classes);
     Scores(:,i) = score(:,2);
     macro_F_scores(i)=F1_macro;
     all_predictions(i,:)=predictions;    
end
   [~,maxScore] = max(Scores,[],2);
   a=sum(~(full(test_class)~=maxScore))/size(maxScore,1);
   accuracies(j)=a;   
   F1_score_micro=get_F1_micro_score(all_predictions,test_class,labels);
   F1_micro_scores(j)=F1_score_micro;
   F1_macro_scores(j)=mean(macro_F_scores);
   results_per_sample(j,:)=macro_F_scores; 
end
accuracies
F1_micro_scores
F1_macro_scores
time_per_sample_size
figure(1)
plot(samples,accuracies,'LineWidth',2)
title('Accuracy of SVM classification')
xlabel('Num samples')
ylabel('Accuracy')
axis([100 800 0 1])

figure(2)
plot(samples,time_per_sample_size,'LineWidth',2)
title('Runtime of ranking data points')
xlabel('Num samples')
ylabel('Time (in seconds)')