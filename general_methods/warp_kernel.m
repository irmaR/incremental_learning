function [ output_args ] = warp_kernel(K,fea,Dist,options )
if isfield(options,'ReguBeta') && options.ReguBeta > 0
    if isfield(options,'W')
        W = options.W;
    else
        if isfield(options,'k')
            Woptions.k = options.k;
        else
            Woptions.k = 0;
        end
        tmpD = Dist;
        nSmp=size(fea,1);
        Woptions.t = options.t;
        if isfield(options,'gnd')
          Woptions.gnd = options.gnd ;
          Woptions.NeighborMode = 'Supervised' ;
        end
        Woptions.WeightMode = 'HeatKernel'  ;
        W = constructW(fea,Woptions);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
    size(L)
    K=(speye(size(K,1))+options.ReguBeta*K*L)\K;
    K = max(K,K');
end

end

