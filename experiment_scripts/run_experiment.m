function [results]=run_experiment(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,method,data_limit,r,warping,blda,k,WeightMode,NeighborMode)
   switch lower(method)
       case {lower('incr')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'incr',r,warping,blda,k,WeightMode,NeighborMode);
       case {lower('incr_bal')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'incr_bal',r,warping,blda,k,WeightMode,NeighborMode);
       case {lower('batch')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'batch',r,warping,blda,k,WeightMode,NeighborMode);
       case {lower('rnd')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'rnd',r,warping,blda,k,WeightMode,NeighborMode);
       case {lower('lssvm')}
          results=incremental_lssvm(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,nr_samples,interval,batch_size,report_points,data_limit,'lssvm',r);
   end
end


function [results]=incremental_lssvm(training_data,training_class,test_data,test_class,kernel_params,gamma_params,nr_samples,interval,batch_size,report_points,data_limit,experiment_name,r)
results=[];   
start_tuning=tic; 
validation_res=zeros(length(kernel_params),length(gamma_params));
k=1;
kernel='RBF_kernel';
   start_tuning=tic;  
   for i=1:length(kernel_params)
     for j=1:length(gamma_params)
         fprintf('HERE')
         options.gamma=gamma_params(j);
         options.kernel=kernel_params(i);
         options.kernel_type=kernel;
         folds=split_into_k_folds(training_data,training_class,5);
                   performances=[];
          
           %go through each fold
         for k=1:length(folds)
            fprintf('Fold: %d',k)
            %increase batch size and interval for optimization
            increment=5;
            if batch_size*increment>=nr_samples
                batch_size_up=nr_samples/2;
            else
                batch_size_up=batch_size*increment;
            end
            interval_up=interval*2;
            
            train_batch=folds{k}.train;
            train_batch_class=folds{k}.train_class;
            report_points_up=[nr_samples:interval_up:size(folds{k}.train,1)-interval_up];
            
            %shuffle the data, splitting into folds might have messed up
            %the things and sorted the data
            s = RandStream('mt19937ar','Seed',run);    
            ix=randperm(s,size(train_batch,1))';
            train_batch=train_batch(ix,:);
            train_batch_class=train_batch_class(ix,:);
            [selected_points,selected_labels,~,~,~]=MAED_experiment_instance(train_batch,train_batch_class,nr_samples,batch_size,options,report_points_up,data_limit,experiment_name,warping);
            aucs=[];
              for s=1:size(selected_points,1)
                  area=run_inference_lssvm(selected_points(s),folds{k}.train,folds{k}.train_class,selected_labels(s),folds{k}.test,folds{k}.test_class,options)
                  %features   = AFEm(selected_points(s),options.kernel_type, options.kernel,X);
                  %features_t = AFEm(selected_points(s),options.kernel_type, options.kernel,folds{k}.test);
                  %[w,b,Yht] = ridgeregress(features,folds{k}.test_class,options.gamm,features_t);
                  %Yht = sign(Yht);
                  aucs(s)=area;
              end
              performances(k)=mean(aucs);
            %end
          end
          area=mean(performances);
          validation_res(i,j,b)=area;
       end
   end
   tuning_time=toc(start_tuning)
   fprintf('Performances')
   %Get best options
   [minp,ic] = max(validation_res,[],1);
   [minminp,is] = max(minp);
   [minmink,is1] = max(minminp);
   ic=ic(is);
   is=is(:,:,is1);
   ic=ic(:,:,is1);
   kernel = kernel_params(ic);
   gamma = gamma_params(is);
   options = [];
   options.kernel_type = 'RBF_kernel';
   options.kernel = kernel;
   options.gamma=gamma;
   sprintf('Run %d, kernel: %f, gamma: %f',run,options.kernel,options.gamma)
   %measure time
   tic;
   %shuffle data
   s = RandStream('mt19937ar','Seed',run);    
   ix=randperm(s,size(training_data,1))';
   training_data=training_data(ix,:);
   training_class=training_class(ix,:);
   [selected_points,selected_labels,~,~,~]=MAED_experiment_instance(training_data,training_class,nr_samples,batch_size,options,report_points_up,data_limit,experiment_name,warping);
   runtime=toc
   best_options=options;
   aucs=[];
   aucs_lssvm=[];
   for k=1:length(selected_kernels)
       Xs=cell2mat(selected_points(k));
       aucs(k)=run_inference_lssvm(Xs,Xs,folds{k}.train_class,cell2mat(selected_labels(k)),test_data,test_class,best_options)
   end
 results.selected_points=selected_points;
 results.selected_labels=selected_labels;
 results.best_options=best_options;
 results.validation_res=validation_res;
 results.kernel=kernel;
 results.gamma=gamma;
 results.aucs=aucs;
 results.tuning_time=tuning_time;
 results.report_points=report_points;
 results.test_points=test_data;
 results.test_labels=test_class;
 results.runtime=runtime;
 fprintf('RESULTS')
end
    
   
function [results]=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,experiment_name,run,warping,blda,kNN,WeightMode,NeighborMode)   
   results=[];
   validation_results={};
   validation_res=zeros(length(reguAlphaParams),length(kernel_params),length(reguBetaParams));
   k=1;
   start_tuning=tic;  
   
   if length(reguAlphaParams)==1 && length(kernel_params)==1 && length(reguBetaParams)==1
      reguAlpha = reguAlphaParams(1);
      kernel_sigma = kernel_params(1);
      regu_beta = reguBetaParams(1);
      tuning_time=0;
   else
   for i=1:length(reguAlphaParams)
     for j=1:length(kernel_params)
       for b=1:length(reguBetaParams)
          options = [];
          options.KernelType = 'Gaussian';
          options.t = kernel_params(j);
          options.bLDA=blda;
          options.ReguType = 'Ridge';
          options.ReguBeta=reguBetaParams(b);
          options.ReguAlpha = reguAlphaParams(i);   
          options.k=kNN;
          options.WeightMode=WeightMode;
          options.NeighborMode=NeighborMode;
          sprintf('Run %d, Alpha: %f, Sigma: %f',run,options.ReguAlpha,options.t)
          list_of_selected_data_points={};
          list_of_selected_labels={};
          list_of_kernels={};
          
          %split training data into 5 folds for tuning the parameters
          folds=split_into_k_folds(training_data,training_class,5);
          performances=[];
          
          %go through each fold
          for k=1:length(folds)
            fprintf('Fold: %d',k)
            %increase batch size and interval for optimization
            increment=5;
            if batch_size*increment>=nr_samples
                batch_size_up=nr_samples/2;
            else
                batch_size_up=batch_size*increment;
            end
            interval_up=interval*2;
            
            train_batch=folds{k}.train;
            train_batch_class=folds{k}.train_class;
            report_points_up=[nr_samples:interval_up:size(folds{k}.train,1)-interval_up];
            
            %shuffle the data, splitting into folds might have messed up
            %the things and sorted the data
            s = RandStream('mt19937ar','Seed',run);    
            ix=randperm(s,size(train_batch,1))';
            train_batch=train_batch(ix,:);
            train_batch_class=train_batch_class(ix,:);
            [selected_points,selected_labels,list_of_selected_times,selected_kernels,list_of_dists]=MAED_experiment_instance(train_batch,train_batch_class,nr_samples,batch_size,options,report_points_up,data_limit,experiment_name,warping);
              aucs=[];
              for s=1:size(selected_kernels,1)
                  area=run_inference(cell2mat(selected_kernels(s)),cell2mat(selected_points(s)),cell2mat(selected_labels(s)),folds{k}.test,folds{k}.test_class,options); 
                  fprintf('Area %f\t',area)
                  aucs(s)=area;
              end
              performances(k)=mean(aucs);
            %end
          end
          area=mean(performances);
          validation_res(i,j,b)=area;
       end
     end
   end
   
   tuning_time=toc(start_tuning)
   fprintf('Performances')
   validation_res

   %Get best options
   [minp,ic] = max(validation_res,[],1);
   [minminp,is] = max(minp);
   [minmink,is1] = max(minminp);
   ic=ic(is);
   is=is(:,:,is1);
   ic=ic(:,:,is1);
   reguAlpha = reguAlphaParams(ic);
   kernel_sigma = kernel_params(is);
   regu_beta = reguBetaParams(is1);
   end
   options = [];
   options.KernelType = 'Gaussian';
   options.t = kernel_sigma;
   options.bLDA=blda;
   options.ReguBeta=regu_beta;
   options.ReguAlpha = reguAlpha;   
   options.k=kNN;
   options.WeightMode=WeightMode;
   options.NeighborMode=NeighborMode;
   sprintf('Run %d, Alpha: %f, Sigma: %f',run,options.ReguAlpha,options.t)
   %measure time
   tic;
   %shuffle data
   s = RandStream('mt19937ar','Seed',run);    
   ix=randperm(s,size(training_data,1))';
   training_data=training_data(ix,:);
   training_class=training_class(ix,:);
   [selected_points,selected_labels,list_of_selected_times,selected_kernels,list_of_dists]=MAED_experiment_instance(training_data,training_class,nr_samples,batch_size,options,report_points,data_limit,experiment_name,warping);
   runtime=toc
   best_options=options;
   aucs=[];
   aucs_lssvm=[];
   for k=1:length(selected_kernels)
       Xs=cell2mat(selected_points(k));
       aucs(k)=run_inference(cell2mat(selected_kernels(k)),Xs,cell2mat(selected_labels(k)),test_data,test_class,best_options);
   end
 results.selected_points=selected_points;
 results.selected_labels=selected_labels;
 results.kernels=selected_kernels;
 results.best_options=best_options;
 results.validation_res=validation_res;
 results.reguAlpha=reguAlpha;
 results.reguBeta=regu_beta;
 results.sigma=kernel_sigma;
 results.aucs=aucs;
 results.tuning_time=tuning_time;
 results.report_points=report_points;
 results.test_points=test_data;
 results.test_labels=test_class;
 results.runtime=runtime;
 fprintf('RESULTS')
end


function [area]=run_inference(kernel,selected_tr_points,selected_tr_labels,test_data,test_class,options)
   %if we only have one class, return area=0
   if length(unique(selected_tr_labels))==1
       fprintf('only one label selected!\n')
       unique(selected_tr_labels)
       fprintf('\n')
       area=0.0;
       return
   end

   options.Kernel=1;
   options.ReguType = 'Ridge';
   options.gnd=selected_tr_labels;
   
   [eigvector, elapseKSR] = KSR_caller(options, kernel);
   if isempty(eigvector)
       options=rmfield(options,'gnd');
       Woptions.gnd = selected_tr_labels ;
       Woptions.t = options.t;
       Woptions.k=options.k;
       Woptions.NeighborMode = options.NeighborMode ;
       Woptions.WeightMode=options.WeightMode;
       W = constructW(selected_tr_points,Woptions);
       options.W=W;
       options.ReducedDim = 1;
       [eigvector, elapseKSR] = KSR_caller(options, kernel);
       options=rmfield(options,'W');
       options=rmfield(options,'ReducedDim');
   end
   %fprintf('Size selected training points %d-%d',size(selected_tr_points,1),size(selected_tr_points,2))
   %fprintf('Size test_data %d-%d',size(test_data,1),size(test_data,1))
   Ktest = constructKernel(test_data, selected_tr_points, options);
   Yhat = Ktest*eigvector;
   if sum(isnan(Yhat))~=0
       area=0;
   else
   [X,Y,T,area] = perfcurve(test_class,Yhat,'1');
   end
   options.Kernel=0;
end

function [area]=run_inference_lssvm(Xs,training_data,training_classes,Ys,test_data,test_class,options)
kernel = 'RBF_kernel';
gamma=1;
%fprintf('sigma %f',options.t)
features = AFEm(Xs,kernel, options.t,Xs);    
try,
  [CostL3, gamma_optimal] = bay_rr(features,Ys,gamma,1);
catch,
  warning('no Bayesian optimization of the regularization parameter');
  gamma_optimal = gamma;
end
[w,b] = ridgeregress(features,Ys,gamma_optimal);
Yh0 = AFEm(Xs,kernel, options.t,test_data)*w+b;
echo off;         
[area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class,['n']);
area
end



