
      fea = rand(50,70);
      options = [];
      options.NeighborMode = 'KNN';
      options.k = 5;
      options.WeightMode = 'HeatKernel';
      options.t = 5;
      W = constructW(fea,options);

      options = [];
      options.W = W;
      options.ReguType = 'Ridge';
      options.ReguAlpha = 0.01;
      options.ReducedDim = 1;

      options.KernelType = 'Gaussian';
      options.t = 5;

      Ktrain = constructKernel(fea,[],options);
      options.Kernel = 1;
      [eigvector] = KSR_caller(options, Ktrain);

      feaTest = rand(3,70);
      Ktest = constructKernel(feaTest,fea,options);
      Y = Ktest*eigvector;  % Y is samples in the KSR subspace
      Y