function []=RCV_experiments(method,path_to_data,path_to_results,path_to_code,nr_runs,nr_samples,batch_size,data_limit,interval,warping,blda,k,WeightMode,NeighborMode)
switch nargin
    case 11
        NeighborModes={'Supervised'};
        WeightModes={'HeatKernel','Cosine'}
        ks=[0];
    case 14
        NeighborModes={NeighborMode};
        WeightModes={WeightMode}
        ks=[k];
end
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(path_to_code))  
load(path_to_data)

% reguBetaParams=[0.01,0.02];
% reguAlphaParams=[0.01,0.02];
% kernel_params=[0.02,0.1];

%reguBetaParams=[0,0.01,0.02];
%reguAlphaParams=[0.01,0.02];
%kernel_params=[0.5,0.1];

reguBetaParams=[0,0.01,0.02,0.04,0.08,0.1,0.2];
reguAlphaParams=[0.01,0.02,0.04,0.2,0.3];
kernel_params=[0.01,0.02,0.04,0.5,1,3,5,10];

for ns=1:length(NeighborModes)
    for ws=1:length(WeightModes)
        for kNN=1:length(ks)

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
        aucs=[];
        tuning_time=[];
        runtime=[];
        res=[];
        for c=1:4
           train=folds{r}.train;
           train_class=folds{r}.train_class;

           %delete later!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           %ix=randperm(size(train,1));
           %train=train(ix(1:500),:);
           %train_class=train_class(ix(1:500),:);
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
           res1=run_experiment(train,train_class,test,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,method,data_limit,r,warping,blda,ks(kNN),WeightModes{ws},NeighborModes{ns})
           aucs(c,:)=res1.aucs;
           selected_labels{c}=res1.selected_labels;
           selected_points{c}=res1.selected_points;
           best_options{c}=res1.best_options;
           tuning_time(c,:)=res1.tuning_time;
           runtime(c,:)=res1.runtime;
        end
        res.selected_labels=selected_labels;
        res.selected_points=selected_points;
        res.all_aucs=aucs;
        res.aucs=mean(aucs);
        res.stdev_aucs=std(aucs);
        res.tuning_time=mean(tuning_time);
        res.stdev_tuning_time=std(tuning_time);
        res.runtime=mean(runtime);
        res.stdev_runtime=std(runtime);
        res.best_options=best_options;
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
        end
    end
end