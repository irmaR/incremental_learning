function [list_of_selected_data_points,list_of_selected_labels,list_of_models]=incremental_experiment_libsvm(samples,batch_size,initial,interval,path_to_result,number_of_runs,path_to_data,train_percentage)

load(path_to_data);
fprintf('Data loaded...')
%number of samples to use
%split the data
K=size(fea,1);
N=size(fea,1);
N=K;
batches=batch_size;
results=[];
end_training=round(train_percentage*N/100)
report_points=[initial:interval:(end_training-batch_size)];
fprintf('Dimensions of data %d-%d',size(fea,1),size(fea,2))


for iter=1:number_of_runs
    s = RandStream('mt19937ar','Seed',iter);
    ix = randperm(s,N)';
    ix=ix(1:K,:);
   
    training_ix=ix(1:end_training);
    test_ix=ix(end_training:length(ix));
    train_fea=fea(training_ix,:);
    train_class=gnd(training_ix,:);
    test_fea=fea(test_ix,:);
    test_class=gnd(test_ix,:);
    labels=full(sort(unique(gnd)));
   
    results_per_sample=zeros(1,length(labels)); % a matrix containing class related F1scores  
    time_per_sample_size=[];

    %go through each label
    class_results=[];
    accuracies=[];

    fprintf('Report points'),report_points
    [list_of_selected_svs,list_of_selected_labels,list_of_selected_times,list_of_models]=incremental_experiment_instance_libsvm(train_fea,train_class,samples,batches,report_points);
end
    