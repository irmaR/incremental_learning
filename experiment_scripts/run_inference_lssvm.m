function [area]=run_inference_lssvm(Xs,training_data,training_classes,Ys,test_data,test_class,options)
gamma=1;
%fprintf('sigma %f',options.t)
features = AFEm(Xs,options.kernel_type, options.kernel,Xs);    
try,
  [CostL3, gamma_optimal] = bay_rr(features,Ys,options.gamma,1);
catch,
  warning('no Bayesian optimization of the regularization parameter');
  gamma_optimal = gamma;
end
[w,b] = ridgeregress(features,Ys,options.gamma);
Yh0 = AFEm(Xs,options.kernel_type, options.kernel,test_data)*w+b;
echo off;         
[area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class);
end
