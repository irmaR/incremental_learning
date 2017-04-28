function [matrix,class]=sparse_matrix_sample(S,gnd,N,K)
   %# using find

idx = find(S);
%# draw 4 without replacement
out=randperm(length(idx));
fourRandomIdx = idx(out(1:N));
%# draw 4 with replacement
%# get row, column values
size(S)
j = colperm(S')'

matrix=S(j(1:N,:),1:K);
y = gnd(randsample(length(gnd),N))
%class=full(gnd(j(1:N,:),:));
%size(class)
%y = datasample(class',N)
hist(y)
