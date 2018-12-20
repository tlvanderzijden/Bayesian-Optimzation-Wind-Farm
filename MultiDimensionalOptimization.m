% This file contains the experiment on Bayesian Optimization (BO) by using Gaussian Processes (GP) executing different acquisition functions (AF's).
% This script can be used for two dimensional input 
% Different tests can be executed:
% Sample function: output of measurement point is sampled from a function (Branin)
% Windtunnel: test in scaled wind farm as described in thesis. Power output of three wind turbines is measured
% FLORIS: FLORIS is a model to estimate the behaviour of wakes in wind farms. A FLORIS model is defined and the power is sampled from the model
% First and second section are used to set settings for the experiment

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
close all;

% make sure the functions used are on the matlab path
addpath('./Tools')
dir = fileparts (mfilename ('fullpath')); 
addpath(genpath('Tools'));
addpath('Models'); 

%The experiment variables are defined in a function. See folder 'Models' for different models/experiments
%varExperiment = FLORIS_two_dimensional;
%varExperiment = Sample_function_two_dimensional;
varExperiment = Windtunnel_two_dimensional;

% what kind of test are we doing? 
typeOfTest = varExperiment.typeOfTest; %Sample Function, Wind Tunnel or FLORIS
% We define variables for the script.
nRuns = 1; % This is the number of full runs we do for every acquisition function. In the end we average over all the results.
nInputs = 50;% This is the number of try-out inputs every acquisition function can try during a full run.
nInitialPoints = 4; %This is the number of random chosen measurement points, before continuing to BO
%rng(7, 'twister'); 

%Options for the windturbines. 
nYawInput = varExperiment.nYawInput; % This is the dimension of the input vector 
nTurbines = 3; %Number of windturbines that can be controlled
turbine1 = 1; %identical numbers of windturbines
turbine2 = 2;
turbine3 = 3; 
yawInRad = 0; %Compute yaw angles in degrees or rads. 


%Options for the optimization
optimizeHyperparameters = 1; %Do we want to tune hyperparameters
startOptimizeHyp = 4; %start optimization of hyperparameters after startOptimizeHyp measurements
nStarts = 3; % When we are optimizing with the multi-start algorithm this is the number of starts per dimension which we get in the multi-start optimization algorithm.
useHyperPrior = 1; 
resetHyp = 0; 

% Options for plotting and data saving
displayPlots = 1; % Do we want to get plots from the script results?
plotHypOverTime = 0; %Do we want to plot change of the hyperparameters during the runs;
plotInstant = 0; %Do we want to display the plot while running the script or at the end of the script?
plotConfPlanes = 'on'; %Do we want to plot the confidence planes for 2-input experiments?  
typeOfPlot = 'contour'; %Do we want to plot surface or contour plot for 2-input experiments?
showProgress = 1; %Do we want to show the progress of the Bayesian Optimization? 
options = optimset('Display','off'); %Do we want to show the progress of optimizing the acquisition function? 
color = addcolours; % We define colors.
saveData = 1; %Do we want to save the data and figures, wind tunnel test are always saved

%Maximum distribution options
maxDistribution = 0; %Do we want to compute the maximum distribution and plot it?
nRounds = 20; % We define the number of rounds for particle maximum distribution
nParticles = 1e4; % We define the number of particles used.
probabilityInterval = 0.95; 

%% Kernel, mean, likelihood
%here a different kernel can be specified
%covfunc = {@covSEard};  
covfunc = { 'covMaternard',3}; 
meanfunc = @meanZero;
%meanfunc = {@meanConst};
likfunc = @likGauss; 
inffunc = @infGaussLik; 

if useHyperPrior == 1
    mu = 3; s2 = 0.5^2;
    pg = {@priorGauss,mu,s2};
    prior.cov= {pg;  pg; pg};
    inffunc = {@infPrior,@infGaussLik,prior};
end
%% Define minimum and maximum space for different types of experiments 
if yawInRad == 0
    xMin = varExperiment.xMin;
    xMax = varExperiment.xMax; 
else
    xMin = deg2rad(varExperiment.xMin);
    xMax = deg2rad(varExperiment.xMax);
end

%% Define trial point mesh
ns = varExperiment.ns;
nsPerDimension = varExperiment.nsPerDimension;
x1Range = linspace(xMin(1),xMax(1),nsPerDimension); % These are the plot points in the dimension of x1.
x2Range = linspace(xMin(2),xMax(2),nsPerDimension); % These are the plot points in the dimension of x2.
[x1Mesh,x2Mesh] = meshgrid(x1Range,x2Range); % We turn the plot points into a grid.
Xs = [reshape(x1Mesh,1,ns);reshape(x2Mesh,1,ns)]; % These are the trial points used for generating plot data.

%% We set up the hyperparameters which are stored in the model
hypStart = varExperiment.hyp; %Hyperparameters are imported from the model in log form. 
hyp = hypStart; 

if strcmp(typeOfTest, 'Sample Function'), sfh = varExperiment.sfh; end % Noise parameter for noise added to sample function measurements
if strcmp(typeOfTest, 'FLORIS'), sfh = varExperiment.sfh; end 
lxNoLog = exp(hyp.gp.cov(1:length(hyp.gp.cov)-1));
lfNoLog = exp(hyp.gp.cov(length(hyp.gp.cov))); 
snNoLog = exp(hyp.gp.lik); 

%% Define acquistion functions and the functions we want to test 
%In the model file of the experiment the parameters of the acquisition function are defined
%The different acquisition functions can be switched on 
UCB.number = 1; %identical number to identify acquisition function
UCB.switchedOn = 1; %swich on: 1 
UCB.nameLong = 'Upper Confidence Bound'; %name used in graph titles
UCB.nameShort = 'UCB'; %name used in legend
PI.number = 2; 
PI.switchedOn = 1; 
PI.nameLong = 'test';
PI.nameShort = 'PI';
EI.number = 3; 
EI.switchedOn = 1;
EI.nameLong = 'test';
EI.nameShort = 'EI';


%% make a matrix with the identical number of the acquisition function in the first column and and the corresponding acquistion hyperparameter in the second column 
acqSwitchedOn = [UCB.switchedOn, PI.switchedOn, EI.switchedOn]; 
afNameShort = {UCB.nameShort; PI.nameShort;  EI.nameShort}; 
afNameLong = {UCB.nameLong; PI.nameLong;  EI.nameLong};
afNumber = {UCB.number; PI.number; EI.number}; 

firstRun = 1; 
for acq = 1:3
    if acqSwitchedOn(1,acq) == 1
        Hyp = varExperiment.hyp.Acq.(afNameShort{acq});
        for k = 1:length(Hyp)
            if firstRun == 1
                indice = 1;
                firstRun = 0;
            else
                indice = size(allAF,1) + 1;
            end
            allAF(indice,2) = Hyp(k,1);
            allAF(indice,1) = afNumber{acq};
        end
    end
end
nAF = size(allAF,1);
%% Create export folder if we want to save data
if saveData == 1
    testDate = strrep(datestr(datetime), ':', '.'); %Date of experiment
    parentDir = fullfile(dir, '\Experiment Data\'); 
    disp('Created folder to save data and figures'); 
    folderName = fullfile(parentDir, strcat(typeOfTest,' experiment-', testDate));
    mkdir(folderName); 
end
%% Initialize Arduino when we are doing a windtunnel test
if strcmp(typeOfTest, 'Windtunnel')
    % settings for arduino
    COM = 3; %arduino COM Port
    windAngle = 0;
    
    comPort = strcat('COM',num2str(COM)); 
    instrreset
    arduinoSerial = serial(comPort,'BAUD', 9600);
    connectToArduino(arduinoSerial);

    % Testing connection with Arduino
    disp('Starting test measurement.');
    [RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents(arduinoSerial, 0, 0, 0)
    disp('Please check the connections. Press any key to continue.');
    pause
    
    % Setting windtunnel windangle conditions
    sendToArduino(arduinoSerial,'adjustWindAngle', windAngle)
end

%% Initialize FLORIS when we are doing a FLORIS test
if strcmp(typeOfTest, 'FLORIS')
    % Instantiate a layout without ambientInflow conditions
    layout = generic_6_turb;
    
    % Use the height from the first turbine type as reference height for theinflow profile
    refheight = layout.uniqueTurbineTypes(1).hubHeight;
    
    % Define an inflow struct and use it in the layout, clwindcon9Turb
    layout.ambientInflow = ambient_inflow_log('PowerLawRefSpeed', 8, 'PowerLawRefHeight', refheight, 'windDirection', deg2rad(0) , 'TI0', .05);
    
    % Make a controlObject for this layout
    controlSet = control_set(layout, 'axialInduction');
    
    % Define subModels
    subModels = model_definition('deflectionModel',      'rans', 'velocityDeficitModel', 'selfSimilar', 'wakeCombinationModel', 'quadraticRotorVelocity', 'addedTurbulenceModel', 'crespoHernandez');   
end

%% Define sample functions when we are doing a sample function test 
if strcmp(typeOfTest, 'Sample Function')
    OptimizationFunction = varExperiment.sampleFunction; % This is the function we want to optimize.
end

%% We set up storage parameters.
randomYaw = zeros(nTurbines, nInitialPoints, nRuns); %variable where we store number of the initalization points
sYaw = zeros(nTurbines, nInputs, nAF, nRuns); % These are the chosen measurement points.
sf = zeros(nInputs, nAF, nRuns); % These are the function values at these measurement points.
sPower = zeros(nInputs, nAF, nRuns); % These are the measured function values (with measurement noise).
sPowerNoiseFree = zeros(nInputs, nAF, nRuns); % These are the measured function values (without measurement noise).
sRecommendations = zeros(nTurbines, nInputs, nAF, nRuns); % These are the recommendations of the optimal yaws made at the end of all the measurements.
sRecommendationValue = zeros(nInputs-nInitialPoints, nAF, nRuns); % These are the values at the given recommendation points (if a sample function is used)
sRecommendationBelievedValue = zeros(nInputs, nAF, nRuns); %These are the powers which the GP beliefs we would get at the recommended yaw angels. 
sLx = zeros(nInputs, nYawInput, nAF, nRuns); %This is a variable to store the hyperparameter lx during BO
sLf = zeros(nInputs, nAF, nRuns); %This is a variable to store the hyperparameter lf during BO
sSn = zeros(nInputs, nAF, nRuns); %This is a variable to store the hyperparameter sn during BO

%% We Display the the experiment parameters
disp('Expirement initiated with parameters:'); 
disp(['Number of input points: ',num2str(nInputs)]); 
disp(['Number of initial inputs: ',num2str(nInitialPoints)]); 
disp(['Number of yaw inputs: ',num2str(nYawInput)]); 
disp(['Type of experiment: ', typeOfTest]); 

%% We start doing the experiment runs.
tic;
for run = 1:nRuns
	% We keep track of how the Bayesian Optimization
    disp(['Starting, ',typeOfTest, ' experiment run ',num2str(run),'/',num2str(nRuns),'. Time passed is ',num2str(toc),' seconds.']);
    
    %Display header in command window
    if strcmp(typeOfTest, 'Windtunnel') || showProgress == 1
        fprintf('%10s %10s %10s %10s %7s %7s %10s %10s %24s %18s %10s %15s \n','Iteration','Yaw_1', 'Yaw_2', 'Yaw_3', 'lx_1', 'lx_2', 'lf', 'sn', 'Initialization/BO','Hyp Likelihood','Power','Time Passed');
        fprintf('-------------------------------------------------------------------------------------------------------------------------------\n', 'Initialization/BO','Power', 'Time Passed');
    end
        
    %We select first measurement points. This is the same for all acquisition functions to reduce random effects. (If one AF gets a lucky first point and another a crappy one, it's kind of unfair.   
    for init = 1:nInitialPoints
        if init == 1 %First initial point is greedy power measurement
            randomYaw(1:nYawInput,init,run) = zeros(nYawInput,1);
        else %Next initial points are chosen random
                randomYaw(1:nYawInput,init,run) = initialPoint(xMin, xMax, nInitialPoints, init-1);
        end
        %Display some information about yaws, hyperparameters, power output, likelihood and time passed
        fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2e | %8.2e | %8.2e | %10.2e | %16s | %8.4f |',init,randomYaw(turbine1,init,run) , randomYaw(turbine2,init,run), randomYaw(turbine3,init,run), lxNoLog , lfNoLog, snNoLog, 'Initialization');
        
        %Measure output dependent of the method
        switch typeOfTest 
            case 'Windtunnel'
                power0(init,:, run) = setYawGetPower(arduinoSerial, randomYaw(:,init,run)); %Measure power output in windtunnel
            case 'Sample Function'
                power0(init,:,run) = OptimizationFunction(randomYaw(1:nYawInput,init,run))+sfh*randn(1,1); %Sample corresponding function value.
            case 'FLORIS' %Sample power output from FLORIS
                if yawInRad == 1
                    florisRunner.controlSet.yawAngleIFArray = randomYaw(:,init,run)';
                else
                    florisRunner.controlSet.yawAngleIFArray = deg2rad(randomYaw(:,init,run))';
                end
                florisRunner = floris(layout, controlSet, subModels);
                florisRunner.run
                f = zeros(1,florisRunner.layout.nTurbs);
                for k = 1:florisRunner.layout.nTurbs
                    f(k) = florisRunner.turbineResults(k).power*1e-06; 
                    fnoise(k) = f(k) +sfh*rand(1,1);
                end
                power0(init,:, run) = sum(fnoise);
                power0NoiseFree(init, :,run) = sum(f); 
        end
        nlml = gp(hyp.gp, inffunc, meanfunc, covfunc , likfunc, randomYaw(1:nYawInput, 1:init ,run)', power0(1:init, :,run)); %Likelihood of the hyperparameters
        fprintf('%16.4f | %8.4f | %13f | \n', nlml, power0(init,:, run), toc);
    end
    fprintf('\n'); 
    
	for iAF = 1:nAF %loop through AF's which are switched on
        if strcmp(typeOfTest, 'Windtunnel') || showProgress == 1
            fprintf('%10s %10s %10s %10s %7s %7s %10s %10s %10s %24s %18s %10s %15s \n','Iteration','Yaw_1', 'Yaw_2', 'Yaw_3', 'lx_1', 'lx_2', 'lf', 'sn', 'AF hyperparameter', 'Initialization/BO','Hyp Likelihood','Power','Time Passed');
            fprintf('-------------------------------------------------------------------------------------------------------------------------------\n', 'Initialization/BO','Power', 'Time Passed');
        end
        
        %We reset the hyperparameters to the original ones for the next run and acquisition function
        hyp.gp.cov= log([lxNoLog; lfNoLog]);
        hyp.gp.lik = log(snNoLog);
        hyp.acq = allAF(iAF,2); 
        
        % We implement the first random measurements that we have already done, and the hyperparameters which are not optimized in the initialization fase
        sYaw(:, 1:nInitialPoints, iAF, run) = randomYaw(:,:,run);
		sPower(1:nInitialPoints, iAF, run) = power0(:,:,run); 
        if strcmp(typeOfTest, 'FLORIS'), sPowerNoiseFree(1:nInitialPoints, iAF, run) =power0NoiseFree(:,:,run);  end
        sLx(1:nInitialPoints,:,iAF, run) = repmat(exp(hyp.gp.cov(1:nYawInput))',[nInitialPoints,1]); %we dont optimize hyperparamters 
        sLf(1:nInitialPoints,iAF, run) = exp(hyp.gp.cov(end));
        sSn(1:nInitialPoints,iAF, run) = exp(hyp.gp.lik(1));
        
        % We calculate the first recommendation of the maximum which would be given, based on the data so far. Highest mean value is chosen.
        AF = @(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc,sYaw(1:nYawInput,1:nInitialPoints,iAF,run)',sPower(1:nInitialPoints,iAF,run),x);
        [yawOpt, powerOpt] = optimizeAcquisitionFunction(AF,xMin , xMax, nStarts);
        
        %We add the recommendation to all recommendations and compute real function value. 
		sRecommendations(1:nYawInput, 1, iAF, run) = yawOpt; %input first measurement
		sRecommendationBelievedValue(1, iAF, run) = powerOpt; %output of AF 
		if strcmp(typeOfTest, 'Sample Function'), sRecommendationValue(1, iAF, run) = OptimizationFunction(yawOpt); end %output of first measurement
        if  strcmp(typeOfTest, 'FLORIS') 
                if yawInRad == 1
                    florisRunner.controlSet.yawAngleIFArray = [yawOpt, 0];
                else
                    florisRunner.controlSet.yawAngleIFArray = deg2rad([yawOpt, 0]);
                end
                florisRunner = floris(layout, controlSet, subModels);
                florisRunner.run
                f = zeros(1,florisRunner.layout.nTurbs);
                for k = 1:florisRunner.layout.nTurbs
                    f(k) = florisRunner.turbineResults(k).power*1e-06; 
                end
                sRecommendationValue(1, iAF, run) = sum(f);
        end
            
        %random measurements incorporated. Now switch to BO and loop through the input points. We start after the initial points. 
		for i = nInitialPoints+1:nInputs 
            %Optimizing hyperparameters
            if optimizeHyperparameters == 1 && i>startOptimizeHyp 
                    hyp.gp = minimize(hyp.gp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF,run ));
                    nlml = gp(hyp.gp, inffunc, meanfunc, covfunc , likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF,run ));
            end
            
            %save new hyperparameters in non log form
            sLx(i,:,iAF, run) = exp(hyp.gp.cov(1:nYawInput))';
            sLf(i,iAF, run) = exp(hyp.gp.cov(end));
            sSn(i,iAF, run) = exp(hyp.gp.lik(1));
            
            [~, powerOpt] = optimizeAcquisitionFunction(@(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc,sYaw(1:nYawInput, 1:i-1 ,iAF,run)',sPower(1:i-1, iAF ),x),xMin , xMax, nStarts);
            switch(allAF(iAF,1))
                case 1
                    AF = @(x)(acqUCB(hyp, inffunc, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF ), x)); 
                case 2
                    AF = @(x)(acqPI(hyp, powerOpt, inffunc, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF ), x)); 
                case 3
                    AF = @(x)(acqEI(hyp, powerOpt, inffunc, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF ), x)); 
            end

            % We run a multi-start optimization on the acquisition function to choose the next measurement point. 
            [yawNext, afMax] = optimizeAcquisitionFunction(AF, xMin, xMax, nStarts);
           
			% We store the selected try-out point, look up the function value and turn it into a measurement.
            sYaw(1:nYawInput, i, iAF, run) = yawNext;

            fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2e | %8.2e | %8.2e | %10.2e | %8.4f | %16s | %8.4f | %8.4f | %8.4f ',i, sYaw(turbine1, i, iAF, run), sYaw(turbine2, i, iAF, run), sYaw(turbine3, i, iAF, run), sLx(i,1, iAF, run) , sLx(i,nYawInput, iAF, run), sLf(i,iAF,run), sSn(i,iAF, run), hyp.acq,'BO')
            %take measurement of new point
            switch typeOfTest
                case 'Sample Function', sPower(i, iAF, run) = OptimizationFunction(sYaw(1:nYawInput, i, iAF, run)) + sfh*randn(1,1);
                case 'Windtunnel', sPower(i, iAF, run) = setYawGetPower(arduinoSerial, sYaw(:, i, iAF, run));
                case 'FLORIS'
                    florisRunner.controlSet.yawAngleIFArray = deg2rad(sYaw(:, i, iAF, run))';
                    florisRunner = floris(layout, controlSet, subModels);
                    florisRunner.run
                    f = zeros(1,florisRunner.layout.nTurbs);
                    for k = 1:florisRunner.layout.nTurbs
                        f(k) = florisRunner.turbineResults(k).power*1e-06; 
                        fnoise(k) = f(k) +sfh*rand(1,1);
                    end
                    sPower(i, iAF, run) = sum(fnoise);
                    sPowerNoiseFree(i, iAF, run) = sum(f); 
            end
            fprintf('%16.4f | %12.4f | %13f | \n',nlml, sPower(i, iAF, run) , toc);
          
            % We calculate the prior distribution for this new point.   
            yawM = sYaw(1:nYawInput, 1:i ,iAF,run)'; %measurements we did so far
            powerM = sPower(1:i, iAF,run); %output of measurements
            
            %We let the algorithm make a recommendation of the input, based on all data so far. This is equal to the highest mean. 
            %We use this to calculate the instantaneous regret.
            [yawOpt, powerOpt] = optimizeAcquisitionFunction(@(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc,yawM,powerM,x),xMin , xMax, nStarts);
    
            %add recommendations of maximum yaw and power to all recommendations
            sRecommendations(1:nYawInput, i, iAF, run) = yawOpt;
            sRecommendationBelievedValue(i, iAF, run) = powerOpt;
            if strcmp(typeOfTest, 'Sample Function') 
                sRecommendationValue(i, iAF, run) = OptimizationFunction(yawOpt); 
            end
            if  strcmp(typeOfTest, 'FLORIS') 
                if yawInRad == 1
                    florisRunner.controlSet.yawAngleIFArray = [yawOpt, 0];
                else
                    florisRunner.controlSet.yawAngleIFArray = deg2rad([yawOpt, 0]);
                end
                florisRunner = floris(layout, controlSet, subModels);
                florisRunner.run
                f = zeros(1,florisRunner.layout.nTurbs);
                for k = 1:florisRunner.layout.nTurbs
                    f(k) = florisRunner.turbineResults(k).power*1e-06; 
                end
                sRecommendationValue(i, iAF, run) = sum(f);
            end
        end 
        
        disp('The recommended yaws are:');
        disp(yawOpt);
        disp(['The estimated power for this yaw is: ',num2str(powerOpt)]);
        
        if  strcmp(typeOfTest, 'Windtunnel') || strcmp(typeOfTest, 'FLORIS')
            if strcmp(typeOfTest, 'Windtunnel')
                finalMax = setYawGetPower(arduinoSerial, sRecommendations(:, end , iAF, run)  );
            else
                florisRunner.controlSet.yawAngleIFArray = deg2rad([yawOpt, 0]);
                florisRunner = floris(layout, controlSet, subModels);
                florisRunner.run
                f = zeros(1,florisRunner.layout.nTurbs);
                for k = 1:florisRunner.layout.nTurbs
                    f(k) = florisRunner.turbineResults(k).power*1e-06;
                end
                finalMax = sum(f);
            end
            disp(['The measured power for this yaw is: ', num2str(finalMax)]);
            measurementDifferencePercentage = (finalMax - powerOpt)/powerOpt*100; 
            disp(['Percentage difference between measured power and estimated max power is: ', num2str(measurementDifferencePercentage),'%']); 
            greedyPower =  sPower(1,iAF,run); 
            powerImprovementPercentage = ((finalMax - greedyPower)/greedyPower*100); 
            disp(['The power improvement for this yaw is: ',num2str(powerImprovementPercentage),'%'])
        end
        
       
        %% If desired, we also generate a plot of the result.
		if displayPlots ~= 0
            plotTimeStep = nInputs;
            sLx(nInputs+1,1) = sLx(nInputs,1);
            sLx(nInputs+1,2) = sLx(nInputs,2);
            sLf(nInputs+1) = sLf(nInputs);
            sSn(nInputs+1) = sSn(nInputs);
            
            hyp.gp.cov(1) = log(sLx(plotTimeStep+1, 1));
            hyp.gp.cov(2) = log(sLx(plotTimeStep+1, 2));
            hyp.gp.cov(3) = log(sLf(plotTimeStep+1));
            hyp.gp.lik = log(sSn(plotTimeStep+1));
            % We start by displaying the Gaussian process resulting from the measurements. We make the calculations for the trial points.
            
            [mPost ,s2Post] = gp(hyp.gp, inffunc, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:plotTimeStep ,iAF,run)' ,sPower(1:plotTimeStep,iAF,run), Xs');
            sPost = sqrt(s2Post);
            mPost = reshape(mPost, size(x1Mesh));
            sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
            
            %% We plot the resulting Gaussian process.
                if strcmp(typeOfTest, 'Sample Function') && iAF == 1 && run == 1
                    fs = zeros(ns,1); % These are the function values for the trial points.
                    for i = 1:ns %compute real function values
                        fs(i) = OptimizationFunction(Xs(:,i));
                    end
                    fsMesh = reshape(fs,nsPerDimension,nsPerDimension);
                    %plot real function
                    figure
                    hold on;
                    grid on;
                    meshc(x1Mesh,x2Mesh,fsMesh);
                    surface(x1Mesh,x2Mesh,fsMesh);
                    colormap('jet');
                    xlabel('x_1');
                    ylabel('x_2');
                    zlabel('Branin function output f(x_1,x_2)');
                    view([20,25]);
                    axis([xMin(1),xMax(1),xMin(2),xMax(2),-350,100])
                    if strcmp(typeOfTest, 'Windtunnel') || saveData == 1
                        %export_fig(fullfile(folderName,'RealFunction.png'),'-transparent');
                        savefig(fullfile(folderName,'RealFunction.fig'));
                    end
                    if plotInstant == 1
                        drawnow;
                    end
                end
                
                figure %plot GP
                hold on;
                grid on;
                xlabel('yaw_1');
                ylabel('yaw_2');
                zlabel('Power Output'); 
                title([afNameShort{allAF(iAF,1)} ' Gaussian Process Plot, run: ', num2str(run)]); %set y-label to name of corresponding acquisition function
                
                makeGPPlotMultidimensional(x1Mesh, x2Mesh, mPost, sPost, sYaw(1:nYawInput,1:plotTimeStep,iAF, run)', sPower(1:plotTimeStep, iAF,run),plotConfPlanes, typeOfPlot); 
             
                if strcmp(typeOfTest, 'Windtunnel') || saveData == 1
                    %export_fig(fullfile(folderName,'GP.png'),'-transparent');
                    fileName = strcat('GP_',num2str(afNameShort{acqSwitchedOn(1,iAF)}),'_run_',num2str(run),'.fig'); 
                    savefig(fullfile(folderName,fileName));
                end
                if plotInstant ==1 
                    drawnow;
                end
                
                acqfig = figure; %plot acquisition function
                hold on;
                grid on;
                xlabel('x_1');
                ylabel('x_2');
                
                [~, powerOpt] = optimizeAcquisitionFunction(@(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc,sYaw(1:nYawInput, 1:plotTimeStep ,iAF,run)',sPower(1:plotTimeStep,iAF,run),x),xMin , xMax, nStarts);
                switch allAF(iAF,1)
                    case 1
                    AF = @(x)(acqUCB(hyp, inffunc, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:plotTimeStep ,iAF,run)', sPower(1:plotTimeStep,iAF,run), x));
                    case 2
                    AF = @(x)(acqPI(hyp, powerOpt, inffunc, meanfunc, covfunc, likfunc,sYaw(1:nYawInput, 1:plotTimeStep ,iAF,run)',sPower(1:plotTimeStep,iAF,run), x));
                    case 3
                    AF = @(x)(acqEI(hyp, powerOpt, inffunc, meanfunc, covfunc, likfunc,sYaw(1:nYawInput, 1:plotTimeStep ,iAF,run)',sPower(1:plotTimeStep,iAF,run), x));
                end
               
                %What would be the next point to sample?
                [yawLast, afLastMax] = optimizeAcquisitionFunction(AF, xMin, xMax, nStarts);
               
                % We calculate the acquisition function values at the plot points and plot those.
                afValues = AF(Xs');
                afValues(afValues < -1e100) = min(afValues(afValues > -1e100)); % We set the insanely small numbers (which may occur when the probability is pretty much zero) to the lowest not-insanely-small number.
                afValues = reshape(afValues, nsPerDimension, nsPerDimension); % We put the result in a square format again.
                
                meshc(x1Mesh,x2Mesh,afValues);
                surface(x1Mesh,x2Mesh,afValues);
                scatter3(yawLast(1), yawLast(2), afLastMax, 'ro', 'filled', 'DisplayName', 'Measurment Points');
                view([20,25]);
                colormap('default');
                title([afNameLong{allAF(iAF,1)}, ' AF, run: ', num2str(run)]);
                colorbar
                if plotInstant == 1
                    drawnow; 
                end
                
                if strcmp(typeOfTest, 'Windtunnel') || saveData == 1
                    %export_fig(fullfile(folderName,'GPandACQ.png'),'-transparent');
                    fileName = strcat('ACQ_',num2str(afNameShort{allAF(iAF,1)}),'_run_',num2str(run),'.fig'); 
                    savefig(fullfile(folderName,fileName));
                end
                
                if maxDistribution == 1 %an experiment with max distribution
                    % We set up a Gaussian process to approximate the measurements, giving us the GP for our examples.
                    SPost = postCovMatrix(hyp.gp.cov, covfunc, sYaw(1:nYawInput,1:plotTimeStep,iAF,run), Xs,snNoLog);
                    %Particle Distribution
                    particleDistribution = particleDistr(mPost, SPost, nRounds,nParticles,ns,xMin, xMax);
                    limitDistribution = limitDistr(mPost, SPost, ns, xMin, xMax);
                    %maxInt = maxInterval(limitDistribution, 0.75, xMin, xMax, Xs); 
                    %[boundYaw1, boundYaw2, probVolume] = max2DIntervalRectangle(limitDistribution, probabilityInterval, xMin, xMax); 
                    figure
                    plotMaxDistribution(limitDistribution, [x1Mesh, x2Mesh] , xMin, xMax, 'Limit Distribution');  
                    
                end
              if strcmp(typeOfTest, 'FLORIS')
                  visTool = visualizer(florisRunner);
                  visTool.plot2dIF;
              end
		end % End of check whether we should make plots.
	end % End of iterating over acquisition functions.
end % End of experiment runs.
disp(['We are done with all the experiments! The time passed is ',num2str(toc),'.']);

%% We save all the data we have generated from wind tunnel test
if strcmp(typeOfTest, 'Windtunnel') || saveData == 1
    save(fullfile(folderName, 'Experiment_Data.mat')); %save experiment data
end
%% When doing experiments with a sample function or FLORIS we can calculate the error and cumulative regret
if strcmp(typeOfTest, 'Sample Function') || strcmp(typeOfTest, 'FLORIS')
    % Now that we're done iterating, we calculate the instantaneous and cumulative regrets. Note that for the first we need to use the recommended points of the GPs (the highest mean) while for the
    % latter we need to use the points that were actually tried out.
    if strcmp(typeOfTest, 'Sample Function')
    [yawOptTrue, powerOptTrue] = optimizeAcquisitionFunction(OptimizationFunction, xMin, xMax, nStarts); % Sometimes this optimization function gives the wrong optimum. When your graphs look odd, try running this block again
    else %FLORIS
        yawOptTrue = [28.5; 16.45]; %Derived from random grid search
        powerOptTrue = 4.28;
    end
    
    meanRecommendationValues = mean(sRecommendationValue, 3);
    meanError = powerOptTrue - meanRecommendationValues;
    meanObtainedValues = mean(sPowerNoiseFree, 3); %mean over runs
    meanObtainedRegret = powerOptTrue - meanObtainedValues;
    meanRegret = cumsum(meanObtainedRegret, 1);

    % We make a plot of the regret over time.
    colors = [color.red;color.blue;color.yellow;color.green;color.grey];
    figure;
    %clf(1);
    hold on;
    grid on;
    for i = 1:nAF
        plot(0:nInputs, [0; meanRegret(:,i)], '-', 'DisplayName', [afNameShort{allAF(i,1)},' hyp: ', num2str(allAF(i, 2))]);
        text(nInputs, meanRegret(end, i),[afNameShort{allAF(i,1)},' hyp: ', num2str(allAF(i, 2))] , 'FontSize', 8); 
    end
    legend('Location', 'SouthEast'); 
    xlabel('Measurement number');
    ylabel('Cumulative regret over time');
    % axis([0,nInputs,0,50]);
    if  saveData == 1
                    export_fig(fullfile(folderName,'CumulativeRegret.png'),'-transparent');
    end
    
    % We make a plot of the error over time.
    figure;
    %clf(2);
    hold on;
    grid on;
    if strcmp(typeOfTest, 'FLORIS'), startError = mean(sPower, 'all'); end
    if strcmp(typeOfTest, 'Sample Function'), startError = mean(fs); end
    for i = 1:nAF
        plot(nInitialPoints+1:nInputs, meanError(nInitialPoints+1:nInputs,i), '-', 'DisplayName', [afNameShort{allAF(i,1)},' hyp: ', num2str(allAF(i, 2))]);
    end
    xlabel('Measurement number');
    ylabel('Recommendation error over time');
    legend
    % axis([0,nInputs,0,0.5]);
    if  saveData == 1
        export_fig(fullfile(folderName,'RecommendationError.png'),'-transparent');
    end
    
    % We display the error in the final recommendations.
    disp(['The true power maximum is: ',num2str(powerOptTrue),'The yaws are: '])
    disp(yawOptTrue); 
    disp('The average final recommendation errors were:');
    disp(meanError(end,:)); 
end