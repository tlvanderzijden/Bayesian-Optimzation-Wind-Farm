function testVariables = Windtunnel_two_dimensional()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%These are the obligatory variables
v.nYawInput = 2;
v.xMin = [-45; -45];
v.xMax = [45; 45];
v.xMinrad = deg2rad(v.xMin);
v.xMaxrad = deg2rad(v.xMax);
v.displayName = 'Windtunnel two yaw input';
v.typeOfTest = 'Windtunnel';
v.nsPerd = 51; 

%load(strcat(dir,'\Variable files\hyperparameters_from_grid_one_input'))
v.lf = 0.04; % This is the output length scale. (1 optimized) (0.5 silly)
v.lx = [18.5; 0.2]; % This is the input length scale. (0.6 optimized) (5 silly)
v.sn = 3e-3;
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

