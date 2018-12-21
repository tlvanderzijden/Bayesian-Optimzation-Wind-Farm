function testVariables = SOWFA_two_dimensional()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%These are the obligatory variables
v.nYawInput = 2;
v.xMin = [-40; -40];
v.xMax = [40; 40];
v.nYawInput = 2; 
v.typeOfTest = 'SOWFA';
v.nsPerDimension = 61; 
v.ns = v.nsPerDimension^2; 

v.propGP.covfunc = {'covMaternard', 3};
v.propGP.meanfunc = @meanZero;
v.propGP.likfunc = @likGauss;
v.propGP.inf = @infGaussLik; 
v.propGP.acqfunc = 2; %EI

%Non log form of hyperparameters
v.lf = 1; % This is the output length scale. (1 optimized) (0.5 silly)
v.lx = [20; 20]; % This is the input length scale. (0.6 optimized) (5 silly)
v.sn = 0.01;

%structure for Rasmussen toolbox
v.hyp.gp.cov= log([v.lx; v.lf]); 
v.hyp.gp.lik = log(v.sn);
v.hyp.gp.mean = [];%log(v.mean); 
%vector of hyperparameters of the acquisition function that need to be tested
%multiple acquisition functions can be tested in one run
v.hyp.Acq.UCB =  []; 
v.hyp.Acq.PI = []; 
v.hyp.Acq.EI = [0.01];
testVariables = v; 


end

