function [sample,class,smpRank,values]=select_sample(data,train_class,nr_samples)
%Actively select 8 examples using MAED
  options = [];
  options.KernelType = 'Gaussian';
  options.t = 0.5;
  options.ReguBeta = 100;
  fprintf('Size of kernel %d',size(data,1))
  [smpRank,values] = MAED(data,nr_samples,options);
  for i = 1:length(smpRank)
      sample(i,:)=data(smpRank(i),:);
      class(i,:)=train_class(smpRank(i),:);
  end
  fprintf('Selected the training points')
end
