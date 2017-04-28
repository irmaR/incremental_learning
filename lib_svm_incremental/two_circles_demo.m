% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab


disp(' This demo illustrates the idea of fixed size LS-SVM. ');
disp(' The program consists of 2 steps. In the former, one ');
disp(' constructs a reduced set of support vectors base on ');
disp(' an apropriate criterion on the data. In this case');
disp(' the measure ''kentropy'' is optimized.');
disp(' ');
disp(' In the latter step, one constructs the implicit mapping');
disp(' to feature space based on the eigenvalue decomposition.');
disp(' A parametric linear regression is executed on the mapped');
disp(' data');
disp(' ');
disp(' To see the used cose, use the call');
disp(' ');
disp('>> type demo_fixedsize ');
disp(' ');
disp(' or ');
disp(' ');
disp('>> edit demo_fixedsize ');
disp(' ');
disp(' A dataset is constructed at first...');
%
% dataset
%clear
figure;
rand('twister',2000);	
[x, y] = GenTwoNoisyCircle(200);
[test, test_class] = GenTwoNoisyCircle(10);
split = y ==1;
split_test=test_class==1;
disp(' The parameters are initialized...');
% initiate values
kernel = 'RBF_kernel';
sigma2=5;
gamma=1;
crit_old=-inf;
Nc=5;
Xs=x(1:Nc,:);
Ys=y(1:Nc,:);
disp(' The optimal reduced set is constructed iteratively: ')
% iterate over data
%
tv = 1;
for tel=1:size(x,1)
  %
  % new candidate set
  %
  Xsp=Xs; Ysp=Ys;
  S=ceil(size(x,1)*rand(1));
  Sc=ceil(Nc*rand(1));
  Xs(Sc,:) = x(S,:);
  Ys(Sc,:) = y(S);
  Ncc=Nc;
  % automaticly extract features and compute entropy
  crit = kentropy(Xs,kernel, sigma2);
  if crit <= crit_old,
    crit = crit_old;
    Xs=Xsp;
    Ys=Ysp;
  else
    crit_old = crit;
    [features,U,lam] = AFEm(Xs,kernel, sigma2,x);
    [w,b,Yh] = ridgeregress(features,y,gamma,features);

    plot(x(split,1),x(split,2),'.k',x(~split,1),x(~split,2),'.b');hold on;
    plot(Xs(:,1),Xs(:,2),'*r');
    xlabel('X1'); ylabel('X2'); 
    title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
    hold off;  drawnow
  end
end

disp(' The parametric linear ridge regression is calculated:');
features = AFEm(Xs,kernel, sigma2,x);    
try,
  [CostL3, gamma_optimal] = bay_rr(features,y,gamma,1);
  fprintf('Optimal gamma obtained')
  gamma_optimal
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
title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
hold off; 
fprintf('AUC-ROC %f\n',area)




