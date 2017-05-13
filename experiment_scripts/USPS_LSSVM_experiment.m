function []=USPS_LSSVM_experiment(method,path_to_data,path_to_results,path_to_code,nr_runs,nr_samples,batch_size,data_limit,interval)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(path_to_code))  
load(path_to_data)


general_output=sprintf('%s/smp_%d/bs_%d/',path_to_results,nr_samples,batch_size);
output_path=sprintf('%s/smp_%d/bs_%d/%s/',path_to_results,nr_samples,batch_size,method);
fprintf('Making folder %s',output_path)
mkdir(output_path)

for r=1:length(folds)
    aucs=[];
    tuning_time=[];
    runtime=[];
    res=[];
    
    for c=1:10
       train=folds{r}.train;
       train_class=folds{r}.train_class;
       test=folds{r}.test;
       test_class=folds{r}.test_class;
       %standardize the training and test data
       train=standardizeX(train);
       test=standardizeX(test);
    
    %for each category in train class we run one learning/inference
    %procedure. We calculate AUCs and we average then
    fprintf('Number of training data points %d-%d, class %d\n',size(train,1),size(train,2),size(train_class,1));
    fprintf('Number of test data points %d-%d\n',size(test,1),size(test,2));
    report_points=[nr_samples:interval:size(train,1)-interval];
    
       train_class(train_class~=c)=-1;
       train_class(train_class==c)=1;
       test_class(test_class~=c)=-1;
       test_class(test_class==c)=1;
       res1=run_experiment(train,train_class,test,test_class,[0.5],[0.01],[],nr_samples,interval,batch_size,report_points,method,data_limit,r,0,0,[],[],[])
       aucs(c,:)=res1.aucs;
       tuning_time(c,:)=res1.tuning_time;
       runtime(c,:)=res1.runtime;
    end
    res.aucs=mean(aucs);
    avg_aucs=mean(aucs);
    stdev=std(aucs);
    res.stdev_aucs=std(aucs);
    res.tuning_time=mean(tuning_time);
    res.stdev_tuning_time=std(tuning_time);
    res.runtime=mean(runtime);
    res.stdev_runtime=std(runtime);
    avg_runtime=mean(runtime);
    std_runtime=std(runtime);
    save(sprintf('%s/auc.mat',output_path),'avg_aucs','stdev','report_points','avg_runtime','std_runtime');
    results{r}=res;
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

