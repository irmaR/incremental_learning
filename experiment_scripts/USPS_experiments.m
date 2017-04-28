function []=USPS_experiments(method,path_to_data,path_to_results,nr_runs,nr_samples,batch_size,data_limit,interval,warping,blda)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath('/Users/irma/Documents/MATLAB/incremental_learning'))  
load(path_to_data)

% reguBetaParams=[0.01,0.02];
% reguAlphaParams=[0.01,0.02];
% kernel_params=[0.02,0.1];

reguBetaParams=[0.02];
reguAlphaParams=[0.02];
kernel_params=[0.1];

% reguBetaParams=[0.01,0.02,0.04,0.08,0.1,0.2];
% reguAlphaParams=[0.01,0.02,0.04,0.2,0.3];
% kernel_params=[0.01,0.02,0.04,0.5,1,3,5,10];
general_output=sprintf('%s/smp_%d/bs_%d/',path_to_results,nr_samples,batch_size);
output_path=sprintf('%s/smp_%d/bs_%d/%s/',path_to_results,nr_samples,batch_size,method);

fprintf('Making folder %s',output_path)
mkdir(output_path)
param_info=sprintf('%s/smp_%d/bs_%d/%s/params.txt',path_to_results,nr_samples,batch_size,method)
fileID = fopen(param_info,'w');


fprintf(fileID,'Beta params=: ');
for i=1:length(reguBetaParams)
   fprintf(fileID,'%1.3f',reguBetaParams(i));
end
fprintf(fileID,'\n');
fprintf(fileID,'Alpha params: ');
for i=1:length(reguAlphaParams)
   fprintf(fileID,'%1.3f',reguAlphaParams(i));
end
fprintf(fileID,'\n');
fprintf(fileID,'Kernel params: ');
for i=1:length(kernel_params)
   fprintf(fileID,'%1.3f',kernel_params(i));
end
fprintf(fileID,'\n')
fprintf(fileID,'Nr runs:%d \n',nr_runs);
fprintf(fileID,'nr_samples:%d \n',nr_samples);
fprintf(fileID,'batch_size:%d \n',batch_size);
fprintf(fileID,'data_limit:%d \n',data_limit);
fprintf(fileID,'interval:%d \n',interval);
fprintf(fileID,'Using warping?:%d \n',warping);
fprintf(fileID,'Using balancing?:%d \n',blda);

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
    fprintf('Number of training data points %d-%d, class %d\n',size(train,1),size(train,2),size(train_class,1));
    fprintf('Number of test data points %d-%d\n',size(test,1),size(test,2));
    report_points=[nr_samples:interval:size(train,1)-interval];
    %we don't use validation here. We tune parameters on training data
    %(5-fold-crossvalidation)
    res=run_experiment(train,train_class,test,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,method,data_limit,r,warping,blda)
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
plot_results(general_output,general_output)
plot_data_imbalance(general_output,[1,2])
