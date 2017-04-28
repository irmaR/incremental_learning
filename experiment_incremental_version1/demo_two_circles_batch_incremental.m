% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab
clear all;


disp(' This demo illustrates the idea of incremental learning of adaptive kernel. ');
%
% dataset
%clear
%figure;
rand('twister',2000);	
[x, y] = GenTwoNoisyCircle(200);
[test, test_class] = GenTwoNoisyCircle(10);
report_points=[5:45:size(x,1)-1];
sigma2=5;
options = [];
options.KernelType = 'Gaussian';
options.t = sigma2;
options.ReguType = 'Ridge';
options.ReguAlpha = 0.02;

[list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,list_of_kernels,list_of_dists]=MAED_experiment_instance(x,y,5,1,options,report_points,5,'batch');

%[list_of_selected_data_points,list_of_selected_labels,list_of_selected_times]=batch_experiment(x,y,5,1,options,report_points,5);
%[list_of_selected_svs,list_of_selected_labels,list_of_selected_times]=incremental_experiment_instance_libsvm(x,y,5,1,report_points);
split = y ==1;
split_test=test_class==1;
disp(' The parameters are initialized...');
% initiate values
kernel = 'RBF_kernel';
gamma=1;
% iterate over data
%
tv = 1;
Aucs_b=[];
for tel=1:length(report_points)
    plot(x(split,1),x(split,2),'.k',x(~split,1),x(~split,2),'.b');hold on;
    Xs=list_of_selected_data_points{1,tel};
    Ys=list_of_selected_labels{1,tel};
    plot(Xs(:,1),Xs(:,2),'*r');
    xlabel('X1'); ylabel('X2'); 
    title(['Selected points for observation %s: ' num2str(report_points(tel))]);
    hold off;  drawnow
    features = AFEm(Xs,kernel, sigma2,x);
    try,
      [CostL3, gamma_optimal] = bay_rr(features,y,gamma,1);
    catch,
      warning('no Bayesian optimization of the regularization parameter');
      gamma_optimal = gamma;
    end
    [w,b] = ridgeregress(features,y,gamma_optimal);
    Yh0 = AFEm(Xs,kernel, sigma2,test)*w+b;
    [Yh0,test_class]
    echo off;         
    [area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class);
    Aucs_b(tel)=area;
end

for k=1:length(report_points)
    fprintf('%d %f\n',report_points(k),Aucs_b(k))
end  
figure(2)
plot(report_points,Aucs_b,'LineWidth',2)
ylim([0 1])
xlabel('#observations');
ylabel('ROC-AUC');

Xs=list_of_selected_data_points{1,length(report_points)};
disp(' The parametric linear ridge regression is calculated:');
features = AFEm(Xs,kernel, sigma2,x);    
try,
  [CostL3, gamma_optimal] = bay_rr(features,y,gamma,3);
catch,
  warning('no Bayesian optimization of the regularization parameter');
  gamma_optimal = gamma;
end
[w,b] = ridgeregress(features,y,gamma_optimal);
Yh0 = AFEm(Xs,kernel, sigma2,test)*w+b
echo off;         
[area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class)
%
% make-a-plot
plot(x(split,1),x(split,2),'.k',x(~split,1),x(~split,2),'b');hold on;
plot(test(split_test,1),test(split_test,2),'k',test(~split_test,1),test(~split_test,2),'*');
plot(Xs(:,1),Xs(:,2),'*r');
xlabel('X'); ylabel('Y'); 
title(['Final model prediction: ']);
hold off; 
fprintf('AUC-ROC %f\n',area)