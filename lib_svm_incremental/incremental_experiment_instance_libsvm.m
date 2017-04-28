function [list_of_selected_svs,list_of_selected_labels,list_of_selected_times]=incremental_experiment_instance_libsvm(train_fea,train_class,model_size,batch,model_observation_points)
tic
fprintf('Dimensions of data %d-%d',size(train_fea,1),size(train_fea,2))
list_of_selected_svs=cell(1, length(model_observation_points));
list_of_selected_labels=cell(1, length(model_observation_points));
list_of_kernels=cell(1, length(model_observation_points));
lists_of_dists=cell(1, length(model_observation_points));
train_fea_incremental=train_fea(1:model_size,:);
train_fea_class_incremental=train_class(1:model_size,:);
gamma=1;
point=1;
for j=0:batch:(size(train_fea,1)-model_size-batch)
    fprintf('Fetching %d - %d\n',model_size+j+1,model_size+j+batch)
    %I added N examples to the training pool
    model_size+j+1,model_size+j+batch
    new_points=train_fea(model_size+j+1:model_size+j+batch,:);
    new_points_class=train_class(model_size+j+1:model_size+j+batch,:);
    train_fea_incremental=[train_fea_incremental;new_points];
    train_fea_class_incremental=[train_fea_class_incremental;new_points_class];
    
    type_of_kernel='RBF_kernel';
    caps = [10 20 50 100 200];
    sig2s = [.1 .2 .5];
    nb = 10;
    for k=1:length(sig2s)
       for t = 1:nb
         sel = randperm(size(train_fea_incremental,1)); 
         svX = train_fea_incremental(sel(1:model_size),:);
         train_fea_updates=train_fea_incremental(sel(1:model_size),:);
         train_fea_class_update=train_fea_class_incremental(sel(1:model_size),:);
         features = AFEm(svX,type_of_kernel,sig2s(k), train_fea_updates);
         try,
           [CostL3, gamma_optimal] = bay_rr(features,train_fea_class_update,gamma,3);
         catch,
           %warning('no Bayesian optimization of the regularization parameter');
           gamma_optimal = gamma;
         end         
         [w,b,Yh] = ridgeregress(features,full(train_fea_class_update),gamma_optimal,features);
         performances(t) = mse(train_fea_class_update - Yh);
       end
       minimal_performances(k) = mean(performances);
    end 
   [minp,ic] = min(minimal_performances,[],1);
   [minminp,is] = min(minp);
   capacity = model_size;
   sig2 =     sig2s(is);
   fprintf('optimal sigma %f',sig2) 
   Xs=train_fea_incremental(1:model_size,:);
   Ys=train_fea_class_incremental(1:model_size,:);
   crit_old=-inf;
   
   for tel=1:size(train_fea_incremental)
      %
      % new candidate set
      %
      Xsp=Xs; Ysp=Ys;
      S=ceil(size(train_fea_class_incremental,1)*rand(1));
      Sc=ceil(model_size*rand(1));
      Xs(Sc,:) = train_fea_incremental(S,:);
      Ys(Sc,:) = train_fea_class_incremental(S);
      Ncc=model_size;

      %
      % automaticly extract features and compute entropy
      %
      crit = kentropy(Xs,type_of_kernel, sig2);

      if crit <= crit_old,
        crit = crit_old;
        Xs=Xsp;
        Ys=Ysp;
      else
        crit_old = crit;
      end
   end
   fprintf('Nr. selected points %d-%d',size(Xs,1),size(Xs,2))
   
    if point<=length(model_observation_points) && model_size+j==model_observation_points(point)
        model_size+j
       list_of_selected_svs{point}=svX;
       list_of_selected_labels{point}=train_fea_class_incremental;
       list_of_selected_times(point)=toc;
       point=point+1;
    end
end 
end