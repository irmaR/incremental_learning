function Yhat = callKSR(Xtest, Xtrain, Ytrain)

	% training options (taken from example.m)

options = [];
options.ReguAlpha = 0.01;
options.ReguType = 'Ridge';
options.gnd = Ytrain;   % groundtruth (flair vector length)
options.KernelType = 'Gaussian';
options.t = 5;

	% Call constructKernel for training data and then run KSR to obtain eigenvector

Ktraining = constructKernel(Xtrain, Xtrain, options);
[eigvector, elapseKSR] = KSR(options, Ytrain, Ktraining);
	
	% Call constructKernel on test data based on training data to obtain a new kernel

Ktest = constructKernel(Xtest, Xtrain, options);

	% multiply kernel by the test data eigenvector to obtain prediction mask

Yhat = Ktest*eigvector;