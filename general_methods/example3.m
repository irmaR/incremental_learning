fea = rand(50,70);
      gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
      options = [];
      options.gnd = gnd;
      options.KernelType = 'Gaussian';
      options.t = 5;
      Ktrain = constructKernel(fea,[],options);
      options.Kernel = 1;

      feaTest = rand(4,70);
      feaTest_class = [ones(1,1);ones(1,1)*2;ones(1,1)*3;ones(1,1)*4];
      
      Ktest = constructKernel(feaTest,fea,options);
      fprintf('Size KTrain %d-%d',size(Ktrain,1),size(Ktrain,2))
      eigvector = KSR_caller(options, Ktrain);
      size(eigvector)
      eigvector
      Ytrain = Ktrain*eigvector;    % Ytrain is training samples in the sparse KSR subspace
      Ytest = Ktest*eigvector;    % Ytest is test samples in the sparse KSR subspace
      Ktest
      TrainL=((Ktrain)^-1)*gnd;
      Prediction=Ktest*TrainL
      
      [w,b] = ridgeregress(Ytest,gnd,gamma_optimal);
      Yh0 = Ytest*w+b
      [area_us,se,thresholds,oneMinusSpec,Sens]=roc(Prediction,gnd);
      area_us