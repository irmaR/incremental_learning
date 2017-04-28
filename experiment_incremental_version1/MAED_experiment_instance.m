function [list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,list_of_kernels,lists_of_dists]=MAED_experiment_instance(train_fea,train_class,model_size,batch,options,model_observation_points,data_limit,experiment_type,warping)
tic
list_of_selected_data_points=cell(1, length(model_observation_points));
list_of_selected_labels=cell(1, length(model_observation_points));
list_of_kernels=cell(1, length(model_observation_points));
lists_of_dists=cell(1, length(model_observation_points));
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
            
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
        case 'incr'
            %fprintf('Train size %d',size(train_fea_incremental,1))
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping);
    end
    % report selected points   
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


function [current_sample,current_labels,ranking,kernel,current_D]=update_model(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
if size(new_points,1)==0
 if size(train_fea_incremental,1)>=data_limit
    ix=randperm(size(train_fea_incremental,1));
    train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
    train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
 end
 [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
 %fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
 current_sample=train_fea_incremental(ranking,:);
 current_labels=train_fea_class_incremental(ranking,:);
 [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
 %fprintf('Current training pool size: %d-%d',size(train_fea_incremental,1))
 %fprintf('Current sample size: %d-%d',size(current_sample,1),size(current_sample,2))
 
else
    selected_samples=train_fea_incremental(ranking,:);
    selected_labels=train_fea_class_incremental(ranking,:);
    samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
    %(size(selected_samples,1)+1),size(new_points,1)
    if (size(selected_samples,1)+1)>size(new_points,1)
        indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:);   
    else
        indices_to_remove=[];
    end
    [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,nr_samples,options,warping);
    %fprintf('Indices to remove',indices_to_remove)
    %fprintf('Kernel size %d',size(kernel,1))
    current_sample=updated_sample;
    current_labels=updated_class;
    %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end
end
