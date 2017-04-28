function []=two_circles_experiment(method,path_to_results,nr_runs,nr_samples,batch_size,data_limit,warping,blda)
%clear all;
%close all;
%clc;
addpath(genpath('/Users/irma/Documents/MATLAB/incremental_learning'))  


% reguBetaParams=[0.01,0.02];
% reguAlphaParams=[0.01,0.02];
% kernel_params=[0.02,0.1];

reguBetaParams=[0.01,0.02,0.04,0.08,0.1,0.2];
reguAlphaParams=[0.01,0.02,0.04,0.2,0.3];
kernel_params=[0.01,0.02,0.04,0.5,1,3,5,10];

%reguAlphaParams=[0.3];
%kernel_params=[0.01];


%reguAlphaParams=[0.01];
%kernel_params=[1];

%nr_samples=50;
%batch_size=1;
output_path=sprintf('%s/smp_%d/bs_%d/%s/',path_to_results,nr_samples,batch_size,method);
fprintf('Making folder %s',output_path)
mkdir(output_path)
interval=5;


for r=1:nr_runs
    s = RandStream('mt19937ar','Seed',r);
    rand('twister',r*1000);	
    [train, train_class] = GenTwoNoisyCircle(100,70,30);
    rand('twister',r*2*1000);
    [test, test_class] = GenTwoNoisyCircle(10,60,40);
    rand('twister',r*3*1000);
    report_points=[nr_samples:batch_size:size(train,1)-batch_size-interval]
    %shuffle the training data
    ix=randperm(s,size(train,1))';
    train=train(ix,:);
    train_class=train_class(ix,:);
    %train=NormalizeFea(train);
    %test=NormalizeFea(test);
    train=standardizeX(train);
    test=standardizeX(test);
    %test=train;
    %test_class=train_class;
    rand('twister',r*4*1000);
    [validation, validation_class] = GenTwoNoisyCircle(10,60,40);
    validation=standardizeX(validation);
    %validation=NormalizeFea(validation);
    res=run_experiment(train,train_class,validation,validation_class,test,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,batch_size,report_points,method,data_limit,r,warping,blda)
    results{r}=res;
end

avg_aucs=zeros(1,length(report_points));
avg_aucs_lssvm=zeros(1,length(report_points));
for i=1:nr_runs
 avg_aucs=avg_aucs+results{i}.aucs;
% avg_aucs_lssvm=avg_aucs_lssvm+results{i}.aucs_lssvm;
end
avg_aucs=avg_aucs/nr_runs;
%avg_aucs_lssvm=avg_aucs_lssvm/nr_runs;
save(sprintf('%s/auc.mat',output_path),'avg_aucs','report_points');
save(sprintf('%s/results.mat',output_path),'results');
