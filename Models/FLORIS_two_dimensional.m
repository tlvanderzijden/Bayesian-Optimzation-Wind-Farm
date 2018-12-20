function testVariables = FLORIS_two_dimensional()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%These are the obligatory variables
v.nYawInput = 2;
v.xMin = [-40; -40];
v.xMax = [40; 40];
v.xMinrad = deg2rad(v.xMin);
v.xMaxrad = deg2rad(v.xMax);
v.displayName = 'FLORIS one yaw input';
v.typeOfTest = 'FLORIS';
v.nsPerDimension = 31; 
v.ns = v.nsPerDimension^2; 

%Non log form of hyperparameters
v.lf = 0.1; % This is the output length scale. (1 optimized) (0.5 silly)
v.lx = [20; 20]; % This is the input length scale. (0.6 optimized) (5 silly)
v.sn = 0.1;
v.sfh = 0.05; %This is the standard deviation of the noise, noise variance (sigma_n) (0.3 optimized)(0.1 silly)
v.mean = 0.05; 

v.propGP.covfunc = {'covMaternard',3};
v.propGP.meanfunc = @meanZero;
v.propGP.likfunc = @likGauss;
v.propGP.inf = @infGaussLik; 

%structure for Rasmussen toolbox
v.hyp.gp.cov= log([v.lx; v.lf]); 
v.hyp.gp.lik = log(v.sn);
v.hyp.gp.mean = [];%log(v.mean); 
v.hyp.Acq.UCB =  [3]; % column vector 
v.hyp.Acq.PI = []; 
v.hyp.Acq.EI = [];

testVariables = v; 
end

