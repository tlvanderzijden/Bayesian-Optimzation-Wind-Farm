function testVariables = Sample_function_two_dimensional()
%Windtunnel_one_dimensional Summary of this class goes here
%   Detailed explanation goes here

%These are the obligatory variables
v.sampleFunction = @(x)(BraninFunction(x));
v.nYawInput = 2;
v.xMin = [-5;0];
v.xMax = [10;15];
v.xMinrad = deg2rad(v.xMin);
v.xMaxrad = deg2rad(v.xMax);
v.displayName = 'Sample Function two input';
v.typeOfTest = 'Sample Function';
v.nsPerDimension = 61; 
v.ns = v.nsPerDimension^2; 

%Non log form of hyperparameters
v.lf = 250; % This is the output length scale. 
v.lx = [4;18]; % This is the input length scale. 
v.sn =  5;
v.sfh = 5; %This is the standard deviation of the noise, noise variance (sigma_n) (0.3 optimized)(0.1 silly)

%structure for Rasmussen toolbox
v.hyp.gp.cov= log([v.lx; v.lf]); 
v.hyp.gp.lik = log(v.sn);
v.hyp.gp.mean = []; 
v.hyp.acq.kappa = 5; 
v.hyp.acq.xiPI = 0.1;
v.hyp.acq.xiEI = 0.1; 


testVariables = v; 
end
