% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab

clc;
close all;
clear all;
disp(' This demo illustrates the idea of incremental learning of adaptive kernel. ');
%
% dataset
%clear
%figure;
r=1;

rand('twister',2000);	
[x, y] = GenTwoNoisyCircle(200);
[test, test_class] = GenTwoNoisyCircle(10);
nr_samples=10;
batch_size=1;
interval=50;
report_points=[10:interval:size(x,1)-batch_size-interval]

s = RandStream('mt19937ar','Seed',r);
rand('twister',r*1000);	
[train, train_class] = GenTwoNoisyCircle(200);
rand('twister',r*2*1000);
[test, test_class] = GenTwoNoisyCircle(10);
rand('twister',r*4*1000);
    
    %shuffle the training data
ix=randperm(s,size(train,1))';
train=train(ix,:);
train_class=train_class(ix,:);
data_limit=150;
options = [];
options.KernelType = 'Gaussian';
options.t = 10;
options.ReguType = 'Ridge';
options.ReguBeta=0.01;
options.ReguAlpha = 0.04;
sigma2=options.t;
%report_points=[nr_samples:20:size(train,1)-batch_size];
x=train;
y=train_class;


%normalize
x=standardizeX(x);
test=standardizeX(test);

[list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,list_of_kernels,list_of_dists]=incremental_experiment_instance(x,y,nr_samples,batch_size,options,report_points);
%[list_of_selected_svs,list_of_selected_labels,list_of_selected_times]=incremental_experiment_instance_libsvm(x,y,5,1,report_points);

split = y ==1;
split_test=test_class==1;

disp(' The parameters are initialized...');
% initiate values
kernel = 'RBF_kernel';
gamma=1;
crit_old=-inf;
Aucs_incr=[];
for tel=1:length(report_points)
    %plot(x(split,1),x(split,2),'.k',x(~split,1),x(~split,2),'.b');hold on;
    Xs=list_of_selected_data_points{1,tel};
    Ys=list_of_selected_labels{1,tel};
    %plot(Xs(:,1),Xs(:,2),'*r');
    %xlabel('X1'); ylabel('X2'); 
    %title(['Selected points for observation %s: ' num2str(report_points(tel))]);
    %hold off;  drawnow
    %Xs,size(Xs)
    features = AFEm(Xs,kernel, sigma2,Xs);
    try,
      [CostL3, gamma_optimal] = bay_rr(features,Ys,gamma,1);
    catch,
      warning('no Bayesian optimization of the regularization parameter');
      gamma_optimal = gamma;
    end
    [w,b] = ridgeregress(features,Ys,gamma_optimal);
    Yh0 = AFEm(Xs,kernel, sigma2,test)*w+b;
    echo off;         
    [area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class,['n']);
    Aucs_incr(tel)=area;
end

for k=1:length(report_points)
    fprintf('%d %f\n',report_points(k),Aucs_incr(k))
end  
% figure(2)
% plot(report_points,Aucs_incr,'LineWidth',2)
% ylim([0 1])
% xlabel('#observations');
% ylabel('ROC-AUC');
% hold off;
%save('AUCS_s_incremental.mat','Aucs_incr')
%save('report_points.mat','report_points')


Xs=list_of_selected_data_points{1,length(report_points)};
Ys=list_of_selected_labels{1,length(report_points)};
Ktraining=list_of_kernels{1,length(report_points)};

size(Ktraining)
options.ReguType = 'Ridge';
options.Kernel=1;
options.gnd=Ys;
% Woptions.gnd = Ys ;
% Woptions.t = sigma2;
% Woptions.NeighborMode = 'Supervised' ;
% W = constructW(Xs,Woptions);
% options.W=W;
% options.ReducedDim = 1;
%Ktraining = constructKernel(Xs,[], options);
[eigvector, elapseKSR] = KSR_caller(options, Ktraining);
Ktest = constructKernel(test, Xs, options);
Ytest = Ktest*eigvector %projection of test data onto the subspace
Ktrain=((Ktraining)^-1)*Ys;
Prediction =Ktest*Ktrain



[w,b] = ridgeregress(eigvector,Ys,13.5131);
Yh0 = Ytest*w+b
[X,Y,T,AUC] = perfcurve(test_class,Ytest,'1')
[area_us,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class,['n']);
[area_us1,se,thresholds,oneMinusSpec,Sens]=roc(Ytest,test_class,['n']);
area_us1
disp(' The parametric linear ridge regression is calculated:');
features = AFEm(Xs,kernel, sigma2,Xs);  
try,
  [CostL3, gamma_optimal] = bay_rr(features,Ys,gamma,1);
catch,
  warning('no Bayesian optimization of the regularization parameter');
  gamma_optimal = gamma;
end
gamma_optimal
[w,b] = ridgeregress(features,Ys,gamma_optimal);
Yh0 = AFEm(Xs,kernel, sigma2,test)*w+b
echo off;         
[area_lssvm,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class,['n']);
fprintf('AUC lssvm: %f, AUC us: %f',area_lssvm,area_us)
%
% make-a-plot
split_test=test_class==1;
figure
hold on;
plot(x(split,1),x(split,2),'.k',x(~split,1),x(~split,2),'.b');
%plot(test(split_test,1),test(split_test,2),'k',test(~split_test,1),test(~split_test,2),'*');
plot(Xs(:,1),Xs(:,2),'*r');
xlabel('X'); ylabel('Y'); 
title(['Final model prediction: ']);
hold off; drawnow
fprintf('\nAUC-ROC %f\n',area_us1)
