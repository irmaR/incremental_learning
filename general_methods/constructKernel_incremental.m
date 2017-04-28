function [K, options] = constructKernel_incremental(D,options)
if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options) 
       error('parameter error!');
   end
end



%=================================================
if ~isfield(options,'KernelType')
    options.KernelType = 'Gaussian';
end

switch lower(options.KernelType)
    case {lower('Gaussian')}        %  e^{-(|x-y|^2)/2t^2}
%         if ~isfield(options,'t')
%             options.t = 1;
%         end
    case {lower('Polynomial')}      % (x'*y)^d
        if ~isfield(options,'d')
            options.d = 2;
        end
    case {lower('PolyPlus')}      % (x'*y+1)^d
        if ~isfield(options,'d')
            options.d = 2;
        end
    case {lower('Linear')}      % x'*y
    otherwise
        error('KernelType does not exist!');
end


%=================================================

switch lower(options.KernelType)
    case {lower('Gaussian')}       
        K = exp(-D/(2*options.t^2));
    case {lower('Polynomial')}     
        K = D.^options.d;
    case {lower('PolyPlus')}     
        K = (D+1).^options.d;
    otherwise
        error('KernelType does not exist!');
end

