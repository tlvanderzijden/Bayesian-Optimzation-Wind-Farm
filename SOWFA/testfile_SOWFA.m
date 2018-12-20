clear all; 
clc; 
addpath(genpath('.\Tools')); 


%dir = startupTools(); 
varBO.nYawInput = 2; 
nInputs = 30;
nInitialPoints = 2; %number of initial (random/greedy) points
nTurbines = 3; 
nYawInput = 2; 
typeOfTest = 'FLORIS' ; %Windtunnel/SOWFA 
OptimizationFunction = @(x)(BraninFunction(x));
%% FLORIS
 % Instantiate a layout without ambientInflow conditions
    layout = generic_6_turb;
    
    % Use the height from the first turbine type as reference height for theinflow profile
    refheight = layout.uniqueTurbineTypes(1).hubHeight;
    
    % Define an inflow struct and use it in the layout, clwindcon9Turb
    layout.ambientInflow = ambient_inflow_log('PowerLawRefSpeed', 8, 'PowerLawRefHeight', refheight, 'windDirection', 0, 'TI0', .05);
    
    % Make a controlObject for this layout
    controlSet = control_set(layout, 'axialInduction');
    
    % Define subModels
    subModels = model_definition('deflectionModel',      'rans', 'velocityDeficitModel', 'selfSimilar', 'wakeCombinationModel', 'quadraticRotorVelocity', 'addedTurbulenceModel', 'crespoHernandez');   

%% Initialize Arduino
if strcmp(typeOfTest, 'FLORIS')
    
end

%%
for i = 1:nInputs
    if strcmp(typeOfTest, 'FLORIS')
        [varBO] = functionSOWFA_test2(varBO); 
        florisRunner.controlSet.yawAngleIFArray = deg2rad(varBO.sYaw(:,i))';
        florisRunner = floris(layout, controlSet, subModels);
        florisRunner.run
        f = zeros(1,florisRunner.layout.nTurbs);
        for k = 1:florisRunner.layout.nTurbs
            f(k) = florisRunner.turbineResults(k).power*1e-06;
        end
        varBO.sPower(i) = sum(f); 
        
    elseif strcmp(typeOfTest, 'Sample')
        [varBO] = functionSOWFA(varBO); 
        varBO.sPower(i) = OptimizationFunction(varBO.sYaw(1:nYawInput,i)); 
    else
        %SOWFA
    end 
    %fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f | %8.4f  | %12.4f | %10.4f \n',i, varBO.sYaw(1, i), varBO.sYaw(2, i), varBO.sYaw(3, i), varBO.hyp.cov(1), varBO.hyp.cov(2), varBO.hyp.lik, varBO.sPower(i) )
end
