function [list_of_selected_data_points,list_of_selected_data_labels]=run_incremental_libsvm(path_to_results,path_to_data,init_k,interval_k,end_k,batch_size,interval,nr_runs)
samples=[init_k:interval_k:end_k];
train_percentage=80;

%interval=400;
for i=1:length(samples)
  initial=samples(i);
  p=sprintf('%s/incremental/smp_%d/bs_%d/',path_to_results,samples(i),batch_size);
  fprintf('Making folder %s',p)
  mkdir(p)
  [list_of_selected_svs,list_of_selected_data_labels,list_of_models]=incremental_experiment_libsvm(samples(i),batch_size,initial,interval,p,nr_runs,path_to_data,train_percentage);
end
end