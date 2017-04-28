%Two circles data with noise
clear all;	
clc;
rand('twister',5489);	
[fea, gnd] = GenTwoNoisyCircle(10);
[test, test_class] = GenTwoNoisyCircle(20);
split = gnd ==1;

%randomize data?
idx=randperm(size(fea,1));
fea=fea(idx,:);
gnd=gnd(idx);
fprintf('Nr data points: %d',size(fea,1))
%Actively select 8 examples using MAED
options = [];
options.KernelType = 'Gaussian';
options.ReguAlpha = 0.01;
options.ReguType = 'Ridge';
options.t = 5;
%[samples,smpRank,values] = select_sample(fea,8)
[list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,list_of_kernels,lists_of_dists]=incremental_experiment_instance(fea,gnd,5,2,options,[5,15])
fprintf('Size of dataset: %d',size(fea,1))
%[list_of_selected_data_points,list_of_selected_labels,list_of_selected_times]=batch_experiment(fea,gnd,5,1,options,[5,950],500)

%[samples,labels]=select_sample_incremental(fea,gnd,200,10)
fprintf('KERNEL')
list_of_kernels{1,2}
samples=full(list_of_selected_data_points{1,2});
labels=full(list_of_selected_labels{1,2});
svm = svmtrain(samples,labels,'ShowPlot',true,'kernel_function','rbf')
Group = svmclassify(svm,test);
shift = svm.ScaleData.shift;
scale = svm.ScaleData.scaleFactor;
x2 = bsxfun(@plus,test,shift);
x2 = bsxfun(@times,test,scale);
sv = svm.SupportVectors;
alphaHat = svm.Alpha;
bias = svm.Bias;
kfun = svm.KernelFunction;
kfunargs = svm.KernelFunctionArgs;

f = kfun(sv,x2,kfunargs{:})'*alphaHat(:) + bias;
f = -f;
[X,Y,T,AUC] = perfcurve(Group,f,1);


%d=abs(test_class-Group);
%accuracy=(sum(d(:)==0))/double(length(Group))
accuracy=sum(~(full(test_class)~=Group))/size(Group,1);
a=full(list_of_selected_data_points{1,2});
b=full(list_of_selected_labels{1,2});
samples=[a,b];
figure(1);
plot(fea(:,1),fea(:,2),'*b');
hold on
plot(samples(:,1),samples(:,2),'*r');	

