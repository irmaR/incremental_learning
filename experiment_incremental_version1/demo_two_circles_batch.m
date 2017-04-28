% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab


disp(' This demo illustrates the idea of incremental learning of adaptive kernel. ');
%
% dataset
%clear
figure;
rand('twister',5489);	
[fea, gnd] = GenTwoNoisyCircle(20);
[test, test_class] = GenTwoNoisyCircle(10);
report_points=[5:1:size(x,1)-1];
options = [];
options.KernelType = 'Gaussian';
options.t = 0.5;
options.ReguBeta = 100;

%MAED boils down to TED when ReguBeta = 0;
[smpRank,values,Dist,K] = MAED(fea,5,options,2000);
figure(2);
plot(fea(split,1),fea(split,2),'.k',fea(~split,1),fea(~split,2),'.b');
hold on;
for i = 1:length(smpRank)
  plot(fea(smpRank(i),1),fea(smpRank(i),2),'*r');
  text(fea(smpRank(i),1),fea(smpRank(i),2),['\fontsize{16} \color{red}',num2str(i)]);
end
hold off;
figure(3);
plot(fea(split,1),fea(split,2),'.k',fea(~split,1),fea(~split,2),'.b');
hold on;
for i = 1:length(smpRank)
  plot(fea(smpRank(i),1),fea(smpRank(i),2),'*r');	
  text(fea(smpRank(i),1),fea(smpRank(i),2),['\fontsize{16} \color{red}',num2str(i)]);
end
hold off;

Xs=fea(smpRank,:);
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