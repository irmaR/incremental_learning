function [area]=inference(kernel,selected_tr_points,selected_tr_labels,test_data,test_class,options)
   option.Kernel=1;
   options.ReguType = 'Ridge';
   options.gnd=selected_tr_labels;
   [eigvector, elapseKSR] = KSR_caller(options, kernel);
   if isempty(eigvector)
       fprintf('Empty vector')
   end
   if isempty(eigvector)
       options=rmfield(options,'gnd');
       Woptions.gnd = selected_tr_labels ;
       Woptions.t = options.t;
       Woptions.NeighborMode = 'Supervised' ;
       W = constructW(selected_tr_points,Woptions);
       options.W=W;
       options.ReducedDim = 1;
       [eigvector, elapseKSR] = KSR_caller(options, kernel);
       options=rmfield(options,'W');
       options=rmfield(options,'ReducedDim');
   end
   Ktest = constructKernel(test_data, selected_tr_points, options);
   Yhat = Ktest*eigvector;
   [X,Y,T,area] = perfcurve(test_class,Yhat,'1');
   %[area,se,thresholds,oneMinusSpec,Sens]=roc(Yhat,test_class,['n']);
   
   %model = SRKDATrain(selected_tr_points, selected_tr_labels, options); 
   %[accuracy,area,predictlabel]=SRKDAPredict(test_data, test_class, model,'2');
   %fprintf('-------------888----------------')
   %display(selected_tr_points)
   %display(selected_tr_labels)
   %display(test_data)
   %display(test_class)
   %display(kernel)
   %fprintf('AUC: %f',area)
   %fprintf('--------------888---------------')
   options.Kernel=0;
end




