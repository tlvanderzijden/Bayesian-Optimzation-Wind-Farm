function testVariables = Windtunnel_one_dimensional()
%Windtunnel_one_dimensional Summary of this class goes here
%   Detailed explanation goes here

%These are the obligatory variables
v.nYawInput = 1;
v.xMin = -45;
v.xMax = 45;
v.xMinrad = deg2rad(v.xMin);
v.xMaxrad = deg2rad(v.xMax);
v.displayName = 'Windtunnel one yaw input';
v.typeOfTest = 'Windtunnel';
v.ns = 301; 

v.lf = 67.58; % This is the output length scale. (1 optimized) (0.5 silly)
v.lx = 0.219; % This is the input length scale. (0.6 optimized) (5 silly)
v.sn =  0.0256;
v.sfh = 0.3; %This is the standard deviation of the noise, noise variance (sigma_n) (0.3 optimized)(0.1 silly)

%structure for Rasmussen toolbox
v.hyp.gp.cov= log([v.lx; v.lf]); 
v.hyp.gp.lik = log(v.sn);
v.hyp.gp.mean = []; 
v.hyp.acq.kappa = 5; 
v.hyp.acq.xiPI = 0.1;
v.hyp.acq.xiEI = 0.1; 

testVariables = v; 
end
