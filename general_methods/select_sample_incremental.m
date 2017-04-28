function [samples,labels]=select_sample_incremental(data,classes,nr_samples,batch_size)
%Actively select 8 examples using MAED
  options = [];
  options.KernelType = 'Gaussian';
  options.t = 0.5;
  options.ReguBeta = 100;
  batch=1;
  batches=round((size(data,1)-nr_samples)/batch_size);
  fprintf('Number of batches ~ %d\n',batches)
  run=1; 
  
  
  %initial; we assume the data is shuffled so we take examples in sequence
  samples=data(1:nr_samples,:);
  labels=classes(1:nr_samples,:);
  %get initial sample kernel and distance
  [smpRank_orig,values,D] = MAED(samples,nr_samples,options);
  offset=size(samples,1);
  original_samples=samples;
  samples=samples(smpRank_orig,:);
  labels=labels(smpRank_orig,:);
 
  while(run)
  cutoff=offset+batch_size-1;
  fprintf('offset %d, cutoff %d, size data %d\n',offset,cutoff,size(data,1))
  
  batch_points=[]; %new batch points added in this iteration
  
  if offset<=size(data,1)
      fprintf('Batch: %d out of %d\n',batch,batches)
      if cutoff>=size(data,1)
        new_batch=data(offset:size(data,1),:);   
        new_batch_labels=classes(offset:size(classes,1),:); 
        run=0;
      else
        new_batch=data(offset:cutoff,:);
        new_batch_labels=classes(offset:cutoff,:);
      end
      samples_updated=samples(1:size(samples,1)-batch_size,:);
      indices_to_remove=smpRank_orig((size(samples,1)+1)-batch_size:end,:);
      classes_updated=labels(1:size(labels,1)-batch_size);
      %fprintf('Samples updated size %d, new batch %d',size(samples_updated,1),size(new_batch,1))
      samples=[samples_updated;new_batch];
      labels=[classes_updated;new_batch_labels];
      batch_points=new_batch;
      
  else
      %fprintf('Exceeded data size... Stopping the incremental procedure.')
      run=0;
      break
  end
  
  %in case the batches do not cover all the dataset. We pick some random
  %points from the dataset to have the desired kernel size
  if size(samples,1)<nr_samples
      diff_in_size=nr_samples-size(samples,1);
      %augment with random n points from dataset
      extra_samples_indices=randperm(size(data,1));
      extra_samples_indices(1:diff_in_size)
      extra_samples=data(extra_samples_indices(1:diff_in_size),:);
      samples=[samples;extra_samples];
      labels=[labels;classes(extra_samples_indices(1:diff_in_size),:)];
      batch_points=[new_batch;extra_samples];
  end
  
  fprintf('Number of samples: %d\n',size(original_samples,1))    
  [smpRank_orig,values,D] = MAED_incremental(original_samples,batch_points,indices_to_remove,D,nr_samples,options);
  %smpRank_orig = MAED(samples,nr_samples,options);
  smpRank = smpRank_orig+offset*(nr_samples);
  samples=samples(smpRank_orig,:);
  labels=labels(smpRank_orig,:);
  %take half of the ranked samples
  offset=offset+batch_size+1;
  %fprintf('sample size %d',size(samples,1))
  batch=batch+1;
  %fprintf('Selected %d training points\n',length(smpRank))  
  %smpRank = MAED(data,nr_samples,options);
  end
  fprintf('Finalized the selection...')
end


  
  