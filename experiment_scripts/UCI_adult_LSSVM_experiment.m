function []=UCI_adult_LSSVM_experiments(method,path_to_data,path_to_results,path_to_code,nr_runs,nr_samples,batch_size,data_limit,interval)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(path_to_code))  
load(path_to_data)

general_output=sprintf('%s/smp_%d/bs_%d/',path_to_results,nr_samples,batch_size);
output_path=sprintf('%s/smp_%d/bs_%d/%s/',path_to_results,nr_samples,batch_size,method);
fprintf('Making folder %s',output_path)
mkdir(output_path)

for r=1:nr_runs
    s = RandStream('mt19937ar','Seed',r);    
    load(path_to_data)
    %shuffle the training data with the seed according to the run
    ix=randperm(s,size(train,1))';
    %pick 60% of the data in this run to be used
    train=train(ix(1:ceil(size(ix,1)*2/3)),:);
    train_class=train_class(ix(1:ceil(size(ix,1)*2/3),:));
    
    %standardize the training and test data
    train=standardizeX(train);
    test=standardizeX(test);
    
    %this test dataset is pretty big so we will sample 1000 points in each
    %run
    ix=randperm(s,size(test,1))';
    test=test(ix(1:1000),:);
    test_class=test_class(ix(1:1000),:);
    
    train_class(train_class~=1)=-1;
    train_class(train_class==2)=1;
    test_class(test_class~=1)=-1;
    test_class(test_class==2)=1;
    
    fprintf('Number of training data points %d-%d, class %d\n',size(train,1),size(train,2),size(train_class,1));
    fprintf('Number of test data points %d-%d\n',size(test,1),size(test,2));
    report_points=[nr_samples:interval:size(train,1)-interval];
    fprintf('Number of report points:%d',length(report_points))
    %we don't use validation here. We tune parameters on training data
    %(5-fold-crossvalidation)
    res=run_experiment(train,train_class,test,test_class,[0.5],[0.01],[],nr_samples,interval,batch_size,report_points,method,data_limit,r,0,0,[],[],[])
%res=run_experiment(train,train_class,test,test_class,nr_samples,interval,batch_size,report_points,method,data_limit,r)
    results{r}=res;
    %save intermediate results just in case
    save(sprintf('%s/results.mat',output_path),'results');
    avg_aucs=zeros(1,length(report_points));
    avg_aucs_lssvm=zeros(1,length(report_points));

    for i=1:r
       avg_aucs=avg_aucs+results{i}.aucs;
       all_aucs(i,:)=results{i}.aucs;
       run_times(i,:)=results{i}.runtime+results{i}.tuning_time;
    end
    stdev=std(all_aucs);
    avg_aucs=avg_aucs/r;
    avg_runtime=mean(run_times);
    std_runtime=std(run_times);
    save(sprintf('%s/auc.mat',output_path),'avg_aucs','stdev','report_points','avg_runtime','std_runtime');
%    plot_results(general_output,general_output)
end

avg_aucs=zeros(1,length(report_points));
avg_aucs_lssvm=zeros(1,length(report_points));

for i=1:nr_runs
  avg_aucs=avg_aucs+results{i}.aucs;
  all_aucs(i,:)=results{i}.aucs;
  run_times(i,:)=results{i}.runtime+results{i}.tuning_time;
end
stdev=std(all_aucs);
avg_aucs=avg_aucs/nr_runs;
avg_runtime=mean(run_times);
std_runtime=std(run_times);
%avg_aucs_lssvm=avg_aucs_lssvm/nr_runs;
save(sprintf('%s/auc.mat',output_path),'avg_aucs','stdev','report_points','avg_runtime','std_runtime');
save(sprintf('%s/results.mat',output_path),'results');
%plot the result
plot_results(general_output)
%plot_data_imbalance(general_output,[1,2])
end

