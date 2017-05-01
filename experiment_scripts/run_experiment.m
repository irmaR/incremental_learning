function [results]=run_experiment(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,method,data_limit,r,warping,blda)
   
   switch lower(method)
       case {lower('incr')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'incr',r,warping,blda);
       case {lower('batch')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'batch',r,warping,blda);
       case {lower('rnd')}
          results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'rnd',r,warping,blda);
   end
end


%function [results]=select_random_sample(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,experiment_name,r,warping,blda)
    
   
function [results]=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,experiment_name,run,warping,blda)   
   results=[];
   kernel = 'RBF_kernel';
   gamma=1;
   validation_results={};
   validation_res=zeros(length(reguAlphaParams),length(kernel_params),length(reguBetaParams));
   k=1;
   tic  
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
          sprintf('Run %d, Alpha: %f, Sigma: %f',run,options.ReguAlpha,options.t)

          list_of_selected_data_points={};
          list_of_selected_labels={};
          list_of_kernels={};
          
    
          %split training data into 5 folds
          folds=split_into_k_folds(training_data,training_class,5);
          performances=[];
          
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
            
            %if strcmp(experiment_name,'batch')
               %if more samples than data limit, sample data_limit data
               %points
               train_batch=folds{k}.train;
               train_batch_class=folds{k}.train_class;
               if size(folds{k}.train,1)>=data_limit
                   ix=randperm(data_limit);
                   [ranking,values,current_D,Kernel] = MAED(train_batch(ix,:),train_batch_class(ix,:),nr_samples,options,data_limit,warping);
                   Xs=train_batch(ix,:);
                   Ys=train_batch_class(ix,:);
               else
                   [ranking,values,current_D,Kernel] = MAED(train_batch,train_batch_class,nr_samples,options,data_limit,warping);
                   Xs=train_batch;
                   Ys=train_batch_class;
               end
            area=run_inference(Kernel,Xs,Ys,folds{k}.test,folds{k}.test_class,options);  
            performances(k)=area;
          end
          area=mean(performances);
          validation_res(i,j,b)=area;
       end
     end
   end
   tuning_time=toc;
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
   options = [];
   options.KernelType = 'Gaussian';
   options.t = kernel_sigma;
   options.bLDA=blda;
   options.ReguBeta=regu_beta;
   options.ReguAlpha = reguAlpha;      
   sprintf('Run %d, Alpha: %f, Sigma: %f',run,options.ReguAlpha,options.t)
   %measure time
   tic;
   [selected_points,selected_labels,list_of_selected_times,selected_kernels,list_of_dists]=MAED_experiment_instance(training_data,training_class,nr_samples,batch_size,options,report_points,data_limit,experiment_name,warping);
   runtime=toc;
   best_options=options;
   aucs=[];
   aucs_lssvm=[];
   for k=1:length(selected_kernels)
       Xs=cell2mat(selected_points(k));
       size(selected_kernels(k));
       size(Xs);
       size(selected_labels(k));
       aucs(k)=run_inference(cell2mat(selected_kernels(k)),Xs,cell2mat(selected_labels(k)),test_data,test_class,best_options);
   end
 results.selected_points=selected_points;
 results.selected_labels=selected_labels;
 results.kernels=selected_kernels;
 results.best_options=best_options;
 results.AUC=area;
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
   option.Kernel=1;
   options.ReguType = 'Ridge';
   options.gnd=selected_tr_labels;
   
   [eigvector, elapseKSR] = KSR_caller(options, kernel);
   if isempty(eigvector)
       options=rmfield(options,'gnd');
       Woptions.gnd = selected_tr_labels ;
       Woptions.t = options.t;
       Woptions.NeighborMode = 'Supervised' ;
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



