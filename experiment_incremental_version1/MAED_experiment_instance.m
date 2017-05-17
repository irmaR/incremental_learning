function [list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,list_of_kernels,lists_of_dists,lists_of_areas]=MAED_experiment_instance(train_fea,train_class,model_size,batch,options,model_observation_points,data_limit,experiment_type,warping)
tic
list_of_selected_data_points=cell(1, length(model_observation_points));
list_of_selected_labels=cell(1, length(model_observation_points));
list_of_kernels=cell(1, length(model_observation_points));
lists_of_dists=cell(1, length(model_observation_points));
lists_of_areas=cell(1, length(model_observation_points));

%
%classes=unique(train_class);
%ix1=find(train_class==classes(1));
%ix2=find(train_class==classes(2));  
    
%nr_samples1=ceil(model_size/2);
%nr_samples2=model_size-nr_samples1;

%train_fea_incremental=[train_fea(ix1(1:nr_samples1),:);train_fea(ix2(1:nr_samples2),:)];
%train_fea_class_incremental=[train_class(ix1(1:nr_samples1),:);train_class(ix2(1:nr_samples2),:)];
ix=randperm(size(train_fea,1));
train_fea=train_fea(ix,:);
train_class=train_class(ix,:);
train_fea_incremental=train_fea(1:model_size,:);
train_fea_class_incremental=train_class(1:model_size,:);
point=1;
if strcmp(experiment_type,'lssvm')
    current_sample=train_fea_incremental;
    current_labels=train_fea_class_incremental;
    current_area=0.5;
else
[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,size(train_fea_incremental,1),options,data_limit,warping);
current_sample=train_fea_incremental;
current_labels=train_fea_class_incremental;
current_Dists=current_D;
%save current point
current_area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options); 
list_of_kernels{point}=kernel;
lists_of_dists{point}=current_Dists;
end
list_of_selected_data_points{point}=current_sample;
list_of_selected_labels{point}=current_labels;
list_of_selected_times(point)=toc;
lists_of_areas{point}=current_area;

point=point+1;


for j=0:batch:(size(train_fea,1)-model_size-batch)
    %fprintf('Fetching %d - %d\n',model_size+j+1,model_size+j+batch)
    %fprintf('Batch %d',j)
    %fprintf('iter %d',model_size+j-batch)
    macro_F_scores=[];
    %taking new points from the training pool
    new_points=train_fea(model_size+j+1:model_size+j+batch,:);
    new_classes=train_class(model_size+j+1:model_size+j+batch,:);
    %fprintf('Size of new points %d',size(new_points,1))
    %fprintf('Batch: %d',batch)
    
    %Batch or incremental experiment
    switch(experiment_type)
        case 'batch'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            new_points=[];
            new_classes=[];
            old_sample=current_sample;
            old_labels=current_labels;
            old_kernel=kernel;
            old_dists=current_Dists;
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping,batch);
            area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options); 
            if area<current_area
               current_sample=old_sample;
               current_labels=old_labels;
               current_kernel=old_kernel;
               current_Dists=old_dists;       
            else
                current_area=area;
            end
        case 'incr'
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %fprintf('New points %d\n',size(new_points,1))
            old_sample=current_sample;
            old_labels=current_labels;
            old_kernel=kernel;
            old_dists=current_Dists;
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
            area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options); 
            if area<current_area
               current_sample=old_sample;
               current_labels=old_labels;
               kernel=old_kernel;
               current_Dists=old_dists;       
            else
                current_area=area;
            end
        case 'incr_bal'
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %fprintf('New points %d\n',size(new_points,1))
            old_sample=current_sample;
            old_labels=current_labels;
            old_kernel=kernel;
            old_dists=current_Dists;
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_balance(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
            %fprintf('Kernel size:%d\n',size(kernel,1))    
            %keep the new model if it improves the auc
            area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options); 
            if area<current_area
               current_sample=old_sample;
               current_labels=old_labels;
               kernel=old_kernel;
               current_Dists=old_dists;       
            else
                current_area=area;
            end
            
        case 'batch_bal'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            new_points=[];
            new_classes=[];
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_balance(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);

            
        case 'rnd'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %[current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            old_sample=current_sample;
            old_labels=current_labels;
            old_kernel=kernel;
            old_dists=current_Dists;
            if options.bLDA
              [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random_balanced(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            else
              [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            end
            area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options); 
            if area<current_area
               current_sample=old_sample;
               current_labels=old_labels;
               kernel=old_kernel;
               current_Dists=old_dists;       
            else
                current_area=area;
            end
            %fprintf('Kernel size:%d\n',size(kernel,1))
        case 'lssvm'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            Nc=model_size;
            Xs=train_fea_incremental(1:Nc,:);
            Ys=train_fea_class_incremental(1:Nc,:);
            crit_old=-inf;
            for tel=1:500 %we are really doing them a favor, but it's so slow
                Xsp=Xs; Ysp=Ys;
                S=ceil(size(train_fea_incremental,1)*rand(1));
                Sc=ceil(Nc*rand(1));
                Xs(Sc,:) = train_fea_incremental(S,:);
                Ys(Sc,:) = train_fea_class_incremental(S);
                Ncc=Nc;
                crit = kentropy(Xs,options.kernel_type, options.kernel);
                if crit <= crit_old,
                    crit = crit_old;
                    Xs=Xsp;
                    Ys=Ysp;
               else
                    crit_old = crit;
               end    
            end
            old_sample=current_sample;
            old_labels=current_labels;
            current_sample=Xs;
            current_labels=Ys;
            area=run_inference_lssvm(current_sample,train_fea,train_class,current_labels,options.test,options.test_class,options);
            if area<current_area
               current_sample=old_sample;
               current_labels=old_labels;  
            else
                current_area=area;
            end
    % report selected points 
    end
    if point<=length(model_observation_points) && model_size+j<=model_observation_points(point)
       %fprintf('reporting...')
       list_of_selected_data_points{point}=current_sample;
       list_of_selected_labels{point}=current_labels;
       if ~strcmp(experiment_type,'lssvm')
          list_of_kernels{point}=kernel;
          lists_of_dists{point}=current_Dists;
       end
       list_of_selected_times(point)=toc;
       lists_of_areas{point}=current_area;
       %point=point+1;
    end
    if point<=length(model_observation_points) && model_size+j>=model_observation_points(point)
        point=point+1;
    end
end 
end


function [current_sample,current_labels,ranking,kernel,current_D]=update_model_random(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
    ix=randperm(size(train_fea_incremental,1));
    train_fea_incremental=train_fea_incremental(ix(1:nr_samples),:);
    train_fea_class_incremental=train_fea_class_incremental(ix(1:nr_samples),:);
    [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
    %fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
    current_sample=train_fea_incremental(ranking,:);
    current_labels=train_fea_class_incremental(ranking,:);
    kernel=kernel(ranking,ranking);
    current_D=current_D(ranking,ranking);
    %[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
end
 
function [current_sample,current_labels,ranking,kernel,current_D]=update_model_random_balanced(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
    ix=randperm(size(train_fea_incremental,1));
    train_fea_incremental=train_fea_incremental(ix,:);
    train_fea_class_incremental=train_fea_class_incremental(ix,:);
    classes=unique(train_fea_class_incremental);
    ix1=find(train_fea_class_incremental==classes(1));
    ix2=find(train_fea_class_incremental==classes(2));  
    
    nr_samples1=ceil(nr_samples/2);
    nr_samples2=nr_samples-nr_samples1;
    if nr_samples1>size(ix1,1)
        nr_samples1=size(ix1,1);
        nr_samples2=nr_samples-nr_samples1;
    end
    
    if nr_samples2>size(ix2,1)
        nr_samples2=size(ix2,1);
        nr_samples1=nr_samples-nr_samples2;
    end
    
    train_fea_incremental=[train_fea_incremental(ix1(1:nr_samples1),:);train_fea_incremental(ix2(1:nr_samples2),:)];
    train_fea_class_incremental=[train_fea_class_incremental(ix1(1:nr_samples1),:);train_fea_class_incremental(ix2(1:nr_samples2),:)];
    [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
    %fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
    current_sample=train_fea_incremental(ranking,:);
    current_labels=train_fea_class_incremental(ranking,:);
    %kernel=kernel(ranking,ranking);
    %current_D=current_D(ranking,ranking);
    [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
 end

function [current_sample,current_labels,ranking,kernel,current_D]=update_model(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping,batch)
if size(new_points,1)==0
 if size(train_fea_incremental,1)>=data_limit
    ix=randperm(size(train_fea_incremental,1));
    train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
    train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
 end
 [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
 %fprintf('Current kernel size: %d-%d\n',size(kernel,1),size(kernel,2))
 current_sample=train_fea_incremental(ranking,:);
 current_labels=train_fea_class_incremental(ranking,:);
 %[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
 kernel=kernel(ranking,ranking);
 current_D=current_D(ranking,ranking);
 
else
    if batch<nr_samples
        selected_samples=train_fea_incremental(ranking,:);
        indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:); 
        selected_labels=train_fea_class_incremental(ranking,:);
        samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
    else
        indices_to_remove=[];   
        selected_samples=train_fea_incremental;
        selected_labels=train_fea_class_incremental;
        samples_updated=selected_samples;
    end
    

    [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,nr_samples,options,warping);
    %fprintf('Indices to remove')
    %fprintf('Kernel size %d',size(kernel,1))
    current_sample=updated_sample;
    current_labels=updated_class;
    %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end
end


function [current_sample,current_labels,ranking,kernel,current_D]=update_model_balance(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping,batch)
if size(new_points,1)==0
 if size(train_fea_incremental,1)>=data_limit
    ix=randperm(size(train_fea_incremental,1));
    train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
    train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
 end
 %we assume that it's always binary problem, hence we split the data into
 %two classes
 classes=unique(train_fea_class_incremental);
 ix1=find(train_fea_class_incremental==classes(1));
 ix2=find(train_fea_class_incremental==classes(2));  
 

 %determine how many samples to select from each class
 nr_samples1=ceil(nr_samples/2);
 nr_samples2=nr_samples-nr_samples1;
 
  
 
 train_1=train_fea_incremental(ix1,:);
 train_2=train_fea_incremental(ix2,:);
 class_1=train_fea_class_incremental(ix1,:);
 class_2=train_fea_class_incremental(ix2,:);

 [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
 updated_class=train_fea_class_incremental(ranking,:);
 updated_sample=train_fea_incremental(ranking,:);
 ix_up_class1=find(updated_class==classes(1));
 ix_up_class2=find(updated_class==classes(2));
    
 if nr_samples1>size(ix_up_class1,1)
        nr_samples1=size(ix_up_class1,1);
        nr_samples2=nr_samples-nr_samples1;
    end
    
 if nr_samples2>size(ix_up_class2,1)
        nr_samples2=size(ix_up_class2,1);
        nr_samples1=nr_samples-nr_samples2;
    end
    
  current_sample=[updated_sample(ix_up_class1(1:nr_samples1),:);updated_sample(ix_up_class2(1:nr_samples2),:)];
  current_labels=[updated_class(ix_up_class1(1:nr_samples1),:);updated_class(ix_up_class2(1:nr_samples2),:)];
  [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
else
    if batch<=nr_samples
        selected_samples=train_fea_incremental(ranking,:);
        indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:); 
        selected_labels=train_fea_class_incremental(ranking,:);
        samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
    else
        indices_to_remove=[];   
        selected_samples=train_fea_incremental;
        selected_labels=train_fea_class_incremental;
        samples_updated=selected_samples;
    end
    classes=unique(train_fea_class_incremental);
%     try
%     ix1=find(train_fea_class_incremental==classes(1));
%     catch
%         ix1=[];
%     end
%     try
%     ix2=find(train_fea_class_incremental==classes(2));
%     catch
%         ix2=[];
%     end
 %determine how many samples to select from each class
    nr_samples1=ceil(nr_samples/2);
    nr_samples2=nr_samples-nr_samples1;
    indices_to_remove=[];
    [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,size(train_fea_incremental,1)+size(new_points,1),options,warping);
    
    try
    ix_up_class1=find(updated_class==classes(1));
    catch 
        ix_up_class1=[];
    end
    try
    ix_up_class2=find(updated_class==classes(2));
    catch
        ix_up_class2=[];
    end
    if nr_samples1>size(ix_up_class1,1)
        nr_samples1=size(ix_up_class1,1);
        nr_samples2=nr_samples-nr_samples1;
    end

    if nr_samples2>size(ix_up_class2,1)
        nr_samples2=size(ix_up_class2,1);
        nr_samples1=nr_samples-nr_samples2;
    end
    
    current_sample=[updated_sample(ix_up_class1(1:nr_samples1),:);updated_sample(ix_up_class2(1:nr_samples2),:)];
    current_labels=[updated_class(ix_up_class1(1:nr_samples1),:);updated_class(ix_up_class2(1:nr_samples2),:)];
    [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
    
    %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end
end


