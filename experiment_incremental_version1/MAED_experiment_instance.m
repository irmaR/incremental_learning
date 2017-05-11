function [list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,list_of_kernels,lists_of_dists]=MAED_experiment_instance(train_fea,train_class,model_size,batch,options,model_observation_points,data_limit,experiment_type,warping)
tic
list_of_selected_data_points=cell(1, length(model_observation_points));
list_of_selected_labels=cell(1, length(model_observation_points));
list_of_kernels=cell(1, length(model_observation_points));
lists_of_dists=cell(1, length(model_observation_points));

%
train_fea_incremental=train_fea(1:model_size,:);
train_fea_class_incremental=train_class(1:model_size,:);
%current_sample=train_fea_incremental;
%current_labels=train_fea_class_incremental;
[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,size(train_fea_incremental,1),options,data_limit,warping);
point=1;
current_sample=train_fea_incremental;
current_labels=train_fea_class_incremental;
current_Dists=current_D;

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
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping,batch);
        case 'incr'
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %fprintf('New points %d\n',size(new_points,1))
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
            %fprintf('Kernel size:%d\n',size(kernel,1))
        case 'incr_bal'
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %fprintf('New points %d\n',size(new_points,1))
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_balance(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
            %fprintf('Kernel size:%d\n',size(kernel,1))    
            
        case 'rnd'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            %fprintf('Kernel size:%d\n',size(kernel,1))
        case 'lssvm'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            Nc=nr_samples;
            Xs=train_fea_incremental(1:Nc,:);
            Ys=train_fea_class_incremental(1:Nc,:);

            for tel=1:5*length(train_fea_incremental)
                Xsp=Xs; Ysp=Ys;
                S=ceil(length(train_fea_incremental)*rand(1));
                Sc=ceil(Nc*rand(1));
                Xs(Sc,:) = train_fea_incremental(S,:);
                Ys(Sc,:) = train_fea_class_incremental(S);
                Ncc=Nc;
                crit = kentropy(Xs,kernel, sigma2ent);
  
               if crit <= crit_old,
                    crit = crit_old;
                    Xs=Xsp;
                    Ys=Ysp;
               else
                    crit_old = crit;
    % ridge regression    
                %features   = AFEm(Xs,kernel, sigma2,X);
                %features_t = AFEm(Xs,kernel, sigma2,Xt);
                %[w,b,Yht] = ridgeregress(features,Y,gamma,features_t);
                %Yht = sign(Yht);
               end    
            end
            current_sample=Xs;
            current_labels=Ys;
    % report selected points 
    end
    if point<=length(model_observation_points) && model_size+j<=model_observation_points(point)
       %fprintf('reporting...')
       list_of_selected_data_points{point}=current_sample;
       list_of_selected_labels{point}=current_labels;
       list_of_kernels{point}=kernel;
       %fprintf('Saved kernel of size %d at point: %d\n',size(kernel,1),model_observation_points(point))
       lists_of_dists{point}=current_Dists;
       list_of_selected_times(point)=toc;
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
 n_samples1=ceil(nr_sample/2);
 n_samples2=nr_sample-n_samples1;
 
 train_1=train_fea_incrementa(ix1,:);
 train_2=train_fea_incrementa(ix2,:);
 class_1=train_fea_class_incremental(ix1,:);
 class_2=train_fea_class_incremental(ix2,:);
 
 [ranking1,values1,current_D1,kernel1] = MAED(train_1,class_1,n_samples1,options,data_limit,warping);
 [ranking2,values2,current_D2,kernel2] = MAED(train_2,class_2,n_samples2,options,data_limit,warping);
 %fprintf('Current kernel size: %d-%d\n',size(kernel,1),size(kernel,2))
 current_sample=[train_fea_incremental(ranking1,:);train_fea_incremental(ranking2,:)];
 current_labels=[train_fea_class_incremental(ranking1,:);train_fea_class_incremental(ranking2,:)];
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
    ix1=find(train_fea_class_incremental==classes(1));
    ix2=find(train_fea_class_incremental==classes(2));
   
 %determine how many samples to select from each class
    nr_samples1=ceil(nr_samples/2);
    nr_samples2=nr_samples-nr_samples1;
    indices_to_remove=[];
    [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,size(train_fea_incremental,1)+size(new_points,1),options,warping);
    
    
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
    
    %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end
end


