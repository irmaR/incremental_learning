function [Xstandardized] = standardizeX(X)
% Standardize X using 0-1 scaling
Xstandardized = (X - min(X(:)))/(max(X(:)) - min(X(:)));
