function [D,augmented_sample_for_comparison,gnd] = EuDist2_increm_1(original_samples,original_samples_class,D,new_data_point,new_data_point_class,bSqrt)
%EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
%Matlab matrix operations.
%
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%
%    Examples:
%
%       a = rand(500,10);
%       b = rand(1000,10);
%
%       A = EuDist2(a); % A: 500*500
%       D = EuDist2(a,b); % D: 500*1000
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.0 --November/2005
%
%   Written by Deng Cai (dengcai AT gmail.com)
augmented_sample_for_comparison=[original_samples;new_data_point];
gnd=[original_samples_class;new_data_point_class];
%find Eucl. distance of new samples with remaining samples
dists=EuDist2(new_data_point,original_samples,0);
dist_between_them=EuDist2(new_data_point,[],0);
dists_orig=EuDist2(augmented_sample_for_comparison,[],0);
v1=[dists,dist_between_them];
D2=[D,dists'];
D=[D2;v1];
assert(isequal(dists_orig,D)==1);
%fprintf('Finished adding one example to D')
if bSqrt
   D = sqrt(D);

end