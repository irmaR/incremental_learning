%Ranking on the USPS data set (9298 samples with 256 dimensions)
clear;
load('data/USPS.mat');
%number of samples to use
N=2000;
train_percentage=30
test_percentage=60



indices=randperm(size(fea,1));
end_training=round(train_percentage*size(fea,1)/100)

train_fea=fea(1:end_training,:);
train_class=gnd(1:end_training,:);

test_fea=fea(end_training+1:end_training+1+100,:);
test_class=gnd(end_training+1:end_training+1+100,:);

%fea=fea(indices,:);
%gnd=gnd(indices);
labels=sort(unique(gnd));
%[R,gnd]=sparse_matrix_sample(fea,gnd,2000,200);
%size(R)

%Do one against all evaluation

for i=1:length(labels):
  %modify the classes
  gnd(gnd~=labels(i))=labels(i)+1
%Actively select 8 examples using MAED
  options = [];
  options.KernelType = 'Gaussian';
  options.t = 0.5;
  options.ReguBeta = 100;
  smpRank = MAED(train_fea,200,options);
  fprintf('Selected the training points')
for i = 1:length(smpRank)    
    selected(i,:)=train_fea(smpRank(i),:);
    class(i,:)=train_class(i);
end
selected1=full(selected);
fprintf('Training the logistic regression')
B=mnrfit(selected1,class);

%Perform test
pihat = mnrval(B,test_fea,'model','nominal')
[predictions,accuracy]=get_accuracy(pihat,test_class,labels)
[X,Y]=perfcurve(test_class,predictions,1)
%colormap('hot')
%imagesc(pihat)
%colorbar
%SVMModel = fitcsvm(selected1,class)

