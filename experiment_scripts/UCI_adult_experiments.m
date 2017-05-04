function []=UCI_adult_experiments(method,path_to_data,path_to_results,path_to_code,nr_runs,nr_samples,batch_size,data_limit,interval,warping,blda,k,WeightMode,NeighborMode)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
switch nargin
    case 11
        NeighborModes={'Supervised'};
        WeightModes={'HeatKernel','Cosine'}
        ks=[0];
    case 14
        NeighborModes={NeighborMode};
        WeightModes={WeightMode};
        ks=[k];
end

addpath(genpath(path_to_code))  
load(path_to_data)

reguBetaParams=[0,0.01,0.04,0.08,0.1,0.2];
reguAlphaParams=[0.01,0.04,0.2,0.3];
kernel_params=[0.01,0.04,0.5,1,5,10];

%reguBetaParams=[0,0.01,0.04];
%reguAlphaParams=[0.01,0.04];
%kernel_params=[0.01,0.04];

for ns=1:length(NeighborModes)
    for ws=1:length(WeightModes)
        for kNN=1:length(ks)
    fprintf('%d, %s, %s\n',ks(kNN),WeightModes{ws},NeighborModes{ns})
    if ks(kNN)==0 && strcmp(WeightModes{ws},'Binary') && strcmp(NeighborModes{ns},'kNN')
        continue
    end
    general_output=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/',path_to_results,nr_samples,batch_size,NeighborModes{ns},WeightModes{ws},ks(kNN));
    output_path=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',path_to_results,nr_samples,batch_size,NeighborModes{ns},WeightModes{ws},ks(kNN),method);

    fprintf('Making folder %s',output_path)
    mkdir(output_path)
    param_info=sprintf('%s/params.txt',output_path)
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
    
    %this test dataset is pretty big so we will sample 1000 points in each
    %run
    ix=randperm(s,size(test,1))';
    test=test(ix(1:1000),:);
    test_class=test_class(ix(1:1000),:);
    
    fprintf('Number of training data points %d-%d, class %d\n',size(train,1),size(train,2),size(train_class,1));
    fprintf('Number of test data points %d-%d\n',size(test,1),size(test,2));
    report_points=[nr_samples:interval:size(train,1)-interval];
    fprintf('Number of report points:%d',length(report_points))
    %we don't use validation here. We tune parameters on training data
    %(5-fold-crossvalidation)
    res=run_experiment(train,train_class,test,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,method,data_limit,r,warping,blda,ks(kNN),WeightModes{ws},NeighborModes{ns})
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
plot_data_imbalance(general_output,[1,2])
        end
    end
end