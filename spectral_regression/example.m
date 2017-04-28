function example()

% training
options = [];
options.ReguAlpha = 0.01;
options.ReguType = 'Ridge';
options.gnd = Y;   % groundtruth (flair vector length)
options.KernelType = 'Gaussian';
options.t = 5;
K = constructKernel(X,X,options);
[eigvector, elapseKSR] = KSR(options, Y, K);


% prediction
Ktest = constructKernel(X, X_new, options);  % if you predict a X_new, Ktest = constructKernel(X_new,X,options); 
Yhat = Ktest*eigvector;
     
