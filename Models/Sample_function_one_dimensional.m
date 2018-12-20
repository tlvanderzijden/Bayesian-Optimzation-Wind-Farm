function testVariables = Sample_function_one_dimensional()
%Windtunnel_one_dimensional Summary of this class goes here
%   Detailed explanation goes here

%These are the obligatory variables
v.sampleFunction =  @(x)(cos(3*x) - x.^2/9 + x/6); 
v.nYawInput = 1;
v.xMin = -3;
v.xMax = 3;
v.xMinrad = deg2rad(v.xMin);
v.xMaxrad = deg2rad(v.xMax);
v.displayName = 'Sample Function one input';
v.typeOfTest = 'Sample Function';
v.ns = 301; 

v.lf = 2; % This is the output length scale. (1 optimized) (0.5 silly)
v.lx = 0.6; % This is the input length scale. (0.6 optimized) (5 silly)
v.sn =  0.02;
v.sfh = 0.1; %This is the standard deviation of the noise, noise variance (sigma_n) (0.3 optimized)(0.1 silly)

%structure for Rasmussen toolbox
v.hyp.gp.cov= log([v.lx; v.lf]); 
v.hyp.gp.lik = log(v.sn);
v.hyp.gp.mean = []; 
v.hyp.acq.kappa = 5; 
v.hyp.acq.xiPI = 0.1;
v.hyp.acq.xiEI = 0.1; 

testVariables = v; 
end
