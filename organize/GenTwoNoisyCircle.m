
function [fea, gnd] = GenTwoNoisyCircle(N,perc1,perc2)
%   [fea, gnd] = GenTwoNoisyCircle(N)
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
N1=N*perc1/100;
N2=N-N1;
rA = ones(1,N)*2+0.3*rand(1,N);
rB = ones(1,N)*1.5+0.3*rand(1,N);

thetaPos = pi*(2.*[1:N]./N);
zeroA = [0,0];
zeroB = [0,0];

feaA = [rA.*cos(thetaPos)+zeroA(1); rA.*sin(thetaPos)+zeroA(2)]'; %'
feaB = [rB.*cos(thetaPos)+zeroB(1); rB.*sin(thetaPos)+zeroB(2)]'; %'

fea = [feaB;feaA];
gnd = [ones(N,1);2*ones(N,1)];
