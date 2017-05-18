function []=RCV_LSSVM_experiment(method,path_to_data,path_to_results,path_to_code,nr_runs,nr_samples,batch_size,data_limit,interval)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(path_to_code))  
load(path_to_data)

general_output=sprintf('%s/smp_%d/bs_%d/',path_to_results,nr_samples,batch_size);
output_path=sprintf('%s/smp_%d/bs_%d/%s/',path_to_results,nr_samples,batch_size,method);
fprintf('Making folder %s',output_path)
mkdir(output_path)

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
           %train=train(ix(1:(size(train,1)/2)),:);
           %train_class=train_class(ix(1:(size(train_class,1)/2)),:);
           
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
        avg_aucs=res.aucs;
        stdev=res.stdev_aucs;
        res.tuning_time=mean(tuning_time);
        res.stdev_tuning_time=std(tuning_time);
        res.runtime=mean(runtime);
        res.selection_time=mean(selection_time);
        res.processing_time=mean(processing_time);
        res.stdev_runtime=std(runtime);
        res.best_options=best_options;
        avg_runtime=res.runtime;
        std_runtime=res.stdev_runtime;
        save(sprintf('%s/auc.mat',output_path),'avg_aucs','stdev','report_points','avg_runtime','std_runtime');
        results{r}=res;
    end

    avg_aucs=zeros(1,length(report_points));
    avg_aucs_lssvm=zeros(1,length(report_points));

    for i=1:nr_runs
      avg_aucs=avg_aucs+results{i}.aucs;
      all_aucs(i,:)=results{i}.aucs;
      run_times(i,:)=results{i}.runtime+results{i}.tuning_time;
      processing_times(i,:)=results{i}.processing_time;
      selection_times(i,:)=results{i}.selection_time;
    end
    stdev=std(all_aucs);
    avg_aucs=avg_aucs/nr_runs;
    avg_runtime=mean(run_times);
    std_runtime=std(run_times);
    %avg_aucs_lssvm=avg_aucs_lssvm/nr_runs;
    save(sprintf('%s/auc.mat',output_path),'avg_aucs','stdev','report_points','avg_runtime','std_runtime','processing_times','selection_times');
    save(sprintf('%s/results.mat',output_path),'results');
    %plot the result
    plot_results(general_output)
        end

