% This file contains the experiment on Bayesian Optimization (BO) by using Gaussian Processes (GP) executing different acquisition functions (AF's).
% Different inputs can be given 
% nRuns: number of times/runs you want to find a maximum


% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
close all;

addpath('./Tools')
dir = startupTools(); 
realTest = 0; 

%% Initialize Arduino
if realTest > 0
    windAngle = 0;
    
    %% Preparing for connection with Arduino
    % Make sure the baud rate and COM port is same as in Arduino
    % Make sure all connections with the Arduino are cleared up for a clean connection

    instrreset
    arduinoSerial = serial('COM7','BAUD', 9600);
    connectToArduino(arduinoSerial);

    %% Testing connection with Arduino
    disp('Starting test measurement.');
    [RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents(arduinoSerial, 0, 0, 0)
    disp('Please check the connections. Press any key to continue.');
    pause
    
    %% Setting windtunnel initial conditions
    sendToArduino(arduinoSerial, 4, windAngle)
end

%%
% We define colors.
black = [0 0 0];
white = [1 1 1];
red = [0.8 0 0];
green = [0 0.4 0];
blue = [0 0 0.8];
yellow = [0.6 0.6 0];
grey = [0.8 0.8 1];
brown = [0.45 0.15 0.0];                                                      
                        
% We define settings for the script.
nRuns = 1; % This is the number of fu ll runs we do for every acquisition function. In the end we average over all the results.
nInputs = 20; % This is the number of try-out inputs every acquisition function can try during a full run.
nInitialPoints = 2; %This is the number of random chosen measurement points, before continuing to BO
nRounds = 20; % We define the number of rounds for particle maximum distribution
nParticles = 1e4; % We define the number of particles used.
nStarts = 3; % When optimizing an acquisition function, this is the number of starts per dimension which we get in the multi-start optimization algorithm.
nYawInput = 1; % This is the dimension of the input vector.r i = 3:401
nTurbines = 3; %Number of windturbines that can be controlled
turbine1 = 1; %identical numbers of windturbine
turbine2 = 2;
turbine3 = 3; 

saveData = 0; %Do we want to save the data and figures, wind tunnel test are always saved
optimizeHyperparameters = 0; %Do we want to tune hyperparameters
startOptimizeHyp = 6; %start optimization of hyperparameters after startOptimizeHyp measurements
%rng(7, 'twister'); % We fix Matlab's random number generator, so we get the same plots as found online. 
optimizationAlgo = 2;

plotHypOverTime = 0; %Do we want to plot change of the hyperparameters during the runs
displayPlots = 1; % Do we want to get plots from the script results?
plotInstant = 0; %Do we want to display the plot while running the script or at the end of the script?
plotConfPlanes = 'on'; 
showProgress = 1; 
maxDistribution = 1; 

options = optimset('Display','off'); 
disp('Expirement initiated with parameters:'); 
disp(['Number of input points: ',num2str(nInputs)]); 
disp(['Number of initial inputs: ',num2str(nInitialPoints)]); 
disp(['Number of yaw inputs: ',num2str(nYawInput)]); 

if realTest == 1 
    disp('Type of experiment: windtunnel test'); 
else
    disp('Type of experiment: sample function test'); 
end
fprintf('\n');
%% Create export folder if we want to save data
if realTest == 1 || saveData == 1
    testDate = strrep(datestr(datetime), ':', '.'); %Date of experiment
    parentDir = fullfile(dir, '\Experiment Data\'); 
    disp('Created folder to save data and figures'); 
    if saveData == 1 && realTest == 0
        folderName = fullfile(parentDir, strcat('Sample function experiment-', testDate)); 
        mkdir(folderName); %create folder for the experiment data
    else 
        folderName = fullfile(parentDir, strcat('Windtunnel experiment-', testDate));
        mkdir(folderName); 
    end
end
%% Function that we want to optimize
%set xMin and xMax for different experiments, define optimization function and sample optimization function
if realTest == 1
    if nYawInput == 1
        xMin = -45; % This is the minimum yaw input.
        xMax = 45; % This is the maximum yaw input.ns = 301; % If we make plots, how many plot (trial) points do we use?
        ns = 301; % If we make plots, how many plot (trial) points do we use?
        Xs = linspace(xMin, xMax, ns); % These are the trial points used for plots.
    else
        xMin = [-45;-45];
        xMax = [45; 45];
        nsPerDimension = 31; % If we make plots, how many plot (trial) points do we use for each dimension?
        ns = nsPerDimension^2; % This is the number of plot points in total.
        x1Range = linspace(xMin(1),xMax(1),nsPerDimension); % These are the plot points in the dimension of x1.
        x2Range = linspace(xMin(2),xMax(2),nsPerDimension); % These are the plot points in the dimension of x2.
        [x1Mesh,x2Mesh] = meshgrid(x1Range,x2Range); % We turn the plot points into a grid.
        Xs = [reshape(x1Mesh,1,ns);reshape(x2Mesh,1,ns)]; % These are the trial points used for generating plot data.
    end
else
    if nYawInput == 1
        OptimizationFunction = @(x)(cos(3*x) - x.^2/9 + x/6); % This is the function we want to optimize.
        xMin = -3;
        xMax = 3;
        ns = 301; % If we make plots, how many plot (trial) points do we use?
        Xs = linspace(xMin, xMax, ns); % These are the trial points used for plots.
        fs = OptimizationFunction(Xs'); %These are the function values for the trial points.
    else
        OptimizationFunction = @(x)(BraninFunction(x)); % This is the function we want to optimize.
        xMin = [-5;0]; % This is the minimum yaw input for each dimension
        xMax = [10;15] ; % This is the maximum yaw input for each dimension
        nsPerDimension = 31; % If we make plots, how many plot (trial) points do we use for each dimension?
        ns = nsPerDimension^2; % This is the number of plot points in total.
        x1Range = linspace(xMin(1),xMax(1),nsPerDimension); % These are the plot points in the dimension of x1.
        x2Range = linspace(xMin(2),xMax(2),nsPerDimension); % These are the plot points in the dimension of x2.
        [x1Mesh,x2Mesh] = meshgrid(x1Range,x2Range); % We turn the plot points into a grid.
        Xs = [reshape(x1Mesh,1,ns);reshape(x2Mesh,1,ns)]; % These are the trial points used for generating plot data.
        fs = zeros(ns,1); % These are the function values for the trial points.
        for i = 1:ns
            fs(i) = OptimizationFunction(Xs(:,i));
        end
        fsMesh = reshape(fs,nsPerDimension,nsPerDimension);
    end
end

%% Kernel, mean, likelihood
%here a different kernel can be specified without toolbox
%covfunc = {'covMaterniso',3};
covfunc = {@covSEiso}; 
meanfunc = @meanZero;
likfunc = @likGauss;
inf = @infGaussLik; 
%% We set up the hyperparameters
eps = 1e-10; % This is a small number we use for numerical reasons.

if realTest == 1 %if we do test in the wind tunnel the hyperparameters are derived from the grid search
    if nYawInput == 1
        load(strcat(dir,'\Variable files\hyperparameters_from_grid_one_input'))
        lf = lfOpt; % This is the output length scale.
        lx = lxOpt; % This is the input length scale.
        sn = snOpt; 
        sn = 7.1192e-4; 
    else
        lx = 152;
        lf = 0.07;
        sn = 3e-3; 
    end
else
    if nYawInput == 1 
        lf = 1; % This is the output length scale. (1 optimized) (0.5 silly)
        lx = 0.6; % This is the input length scale. (0.6 optimized) (5 silly)
        sn = 0.2; 
        sfh = 0.3; %This is the standard deviation of the noise, noise variance (sigma_n) (0.3 optimized)(0.1 silly)
    else 
        lf = 50; % This is the output length scale. (250 optimized)
        lx = 300; % This is the input length scale. ([4;18] optimized)
        sn= 0.2; % This is the standard deviation of the noise. (5 optimized)
        sfh = 8; 
    end      
end

%structure for toolbox
hyp.gp.cov= [ log(lx) log(lf)]; 
hyp.gp.lik = log(sn);
hyp.gp.mean = []; 

%We set up other acquisition function settings. These have been tuned manually.
kappa = 3;  

if nYawInput == 1
    xiPI = 0.1; % This is the exploration parameter for the PI function.
    xiEI = 0.1; % This is the exploration parameter for the EI function.
else
    xiPI = 2; % This is the exploration parameter for the PI function.
    xiEI = 2; % This is the exploration parameter for the EI function.
end 
%structure
hyp.acq.kappa= kappa;
hyp.acq.xiPI= xiPI;
hyp.acq.xiEI= xiEI;

%BA
gamma = 0.05;
beta = 1.1; 
tauInitial = 0.025 * (xMax-xMin); 

%% which acquisition functions do we want to test?
%First set properties for the AF's 
%first row: identical number of AF
%second row: AF uses code of Bijl (true/false)
%third row: AF switched on (true/false)
UCBs = [1; false; true]; %UCB function self-made
PIs = [2 ;false; false]; %PI function self-made
EIs = [3 ; false; false]; %EI function self-made

allAcq = [UCBs PIs EIs]; %add all ACQ's 
acqSwitchedOn = allAcq([1;2], allAcq(3,:) == true); %row vector containing identical numbers of AF's which are switched on
nAF = size(acqSwitchedOn,2); %number of AF's which are active 

afNameLong = {"Upper Confidence Bound","Probability of Improvement", "Expected Improvement"};
afNameShort = {'UCB', 'PI', 'EI'};

%% We set up storage parameters.
randomYaw = zeros(nTurbines, nInitialPoints, nRuns); %variable where we store number of the initalization points
sYaw = zeros(nTurbines, nInputs, nAF, nRuns); % These are the chosen measurement points.
sf = zeros(nInputs, nAF, nRuns); % These are the function values at these measurement points.
sPower = zeros(nInputs, nAF, nRuns); % These are the measured function values (with measurement noise).
sRecommendations = zeros(nTurbines, nInputs, nAF, nRuns); % These are the recommendations of the optimal yaws made at the end of all the measurements.
sRecommendationValue = zeros(nInputs, nAF, nRuns); % These are the values at the given recommendation points (if a sample function is used)
sRecommendationBelievedValue = zeros(nInputs, nAF, nRuns); %These are the powers which the GP beliefs we would get at the recommended yaw angels. 
sLx = zeros(nInputs, nAF, nRuns); 
sLf = zeros(nInputs, nAF, nRuns); 
sSn = zeros(nInputs, nAF, nRuns); 
%% We start doing the experiment runs.
tic;
for run = 1:nRuns
	% We keep track of how the Bayesian Optimization
    if realTest == 1 || showProgress == 1
        fprintf('%10s %10s %10s %10s %7s %10s %10s %24s %18s %10s %15s \n','Iteration','Yaw_1', 'Yaw_2', 'Yaw_3', 'lx', 'lf', 'sn', 'Initialization/BO','Hyp Likelihood','Power','Time Passed');
        fprintf('-------------------------------------------------------------------------------------------------------------------------------\n', 'Initialization/BO','Power', 'Time Passed');
    end
    
    %We select first measurement points. This is the same for all acquisition functions to reduce random effects. (If one AF gets a lucky first point and another a crappy one, it's kind of unfair.)
    if realTest == 1
		disp(['Starting wind tunnel experiment run ',num2str(run),'/',num2str(nRuns),'. Time passed is ',num2str(toc),' seconds.']);
        
        for init = 1:nInitialPoints
            if init == 1 %First initial point is greedy power measurement
                randomYaw(:,init,run) = zeros(nTurbines,1);
            else %Next initial points are chosen random
                randomYaw(1:nYawInput,init,run) = rand(nYawInput,1).*(xMax-xMin) + xMin;
            end
            fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f | %8.4f | %10.2e | %16s | %8.4f |',init,randomYaw(turbine1,init,run) , randomYaw(turbine2,init,run), randomYaw(turbine3,init,run), lx , lf, sn, 'Initialization'); 
            power0(init,:, run) = setYawGetPower(arduinoSerial, randomYaw(:,init,run));
            nlml = gp(hyp.gp, inf, meanfunc, covfunc , likfunc, randomYaw(1:init, : ,run)', power0(1:init, :,run)); 
            fprintf('%16.4f | %8.4f | %13f | \n', nlml, power0(init,:, run), toc);   
        end
    else
        disp(['Starting test experiment run ',num2str(run),'/',num2str(nRuns),'. Time passed is ',num2str(toc),' seconds.']);
        for init = 1:nInitialPoints
            randomYaw(1:nYawInput,init,run) = rand(nYawInput,1).*(xMax-xMin) + xMin;
            power0(init,:,run) = OptimizationFunction(randomYaw(1:nYawInput,init,run))+sfh*randn(1,1); % This is the corresponding function value. 
            
            if showProgress == 1
                fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f | %8.4f | %10.2e | %16s | %8.4f |',init, randomYaw(turbine1,init,run), randomYaw(turbine2,init,run), randomYaw(turbine3,init,run), lx , lf, sn, 'Initialization');
                nlml = gp(hyp.gp, inf, meanfunc, covfunc , likfunc, randomYaw(1:init, : ,run)', power0(1:init, :,run));
                fprintf('%16.4f | %12.4f | %13f | \n', nlml, power0(init,:, run), toc);
            end
        end
    end
	  
	for iAF = 1:nAF %loop through AF's which are switched on
		% We keep track of how far along we are.
        %disp(['Starting acquisition function ',num2str(iAF),': ', char(afNameLong{acqSwitchedOn(1,iAF)}),'. Time passed is ',num2str(toc), ' seconds.'])   
    
        % We implement the first random measurements that we have already done, and the hyperparameters which are not optimized in the initialization fase
        sYaw(:, 1:nInitialPoints, iAF, run) = randomYaw(:,:,run);
		sPower(1:nInitialPoints, iAF, run) = power0(:,:,run); 
        sLx(1:nInitialPoints,iAF, run) = exp(hyp.gp.cov(1)); %we dont optimize hyperparamters 
        sLf(1:nInitialPoints,iAF, run) = exp(hyp.gp.cov(2));
        sSn(1:nInitialPoints,iAF, run) = exp(hyp.gp.lik(1));
        
        % We calculate the first recommendation of the maximum which would be given, based on the data so far. Highest mean value is chosen.
        if optimizationAlgo == 1
            AF = @(x)acqEV(hyp.gp,@infGaussLik, meanfunc, covfunc,likfunc,randomYaw',power0,x);
            [yawOpt, powerOpt] = optimizeAcquisitionFunction(@(x)acqEV(hyp.gp,@infGaussLik, meanfunc, covfunc,likfunc,randomYaw',power0',x),xMin , xMax, nStarts);
        else
            AF = @(x)acqEV(hyp.gp,@infGaussLik, meanfunc, covfunc,likfunc,sYaw(1:nYawInput,1:nInitialPoints,iAF,run)',sPower(1:nInitialPoints,iAF,run),x);
            [yawOpt] = patternsearch(@(x)(-AF(x)), sYaw(1:nYawInput,1,iAF,run)' ,[],[],[],[], xMin ,xMax, [],options);
            powerOpt = AF(yawOpt);
        end
       
        %We add the recommendation to all recommendations and compute real function value. 
		sRecommendations(1:nYawInput, 1, iAF, run) = yawOpt; %input first measurement
		sRecommendationBelievedValue(1, iAF, run) = powerOpt; %output of AF 
		if realTest ~= 1, sRecommendationValue(1, iAF, run) = OptimizationFunction(yawOpt); end %output of first measurement
        
        % We now start iterating over all the input points. We start from the second point because the first is the same for all acquisition functions anyway. We have already incorporated it.
		for i = nInitialPoints+1:nInputs %random measurements incorporated. Now switch to BO
   
            %Optimizing hyperparameters
            if optimizeHyperparameters == 1 && i>startOptimizeHyp 
                %[sn, lf, lx, mb, logp] = tuneHyperparameters(sYaw(:, 1:i , iAF, run),sPower(1:i, iAF, run));
                hyp.gp = minimize(hyp.gp, @gp, -100, inf, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF,run ));
                nlml = gp(hyp.gp, inf, meanfunc, covfunc , likfunc, sYaw(:, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF,run )); 
            end
            
            %save hyperparameters
            sLx(i,iAF, run) = exp(hyp.gp.cov(1));
            sLf(i,iAF, run) = exp(hyp.gp.cov(2));
            sSn(i,iAF, run) = exp(hyp.gp.lik(1));
            
            switch(acqSwitchedOn(1,iAF))
                case 1
                    AF = @(x)(acqUCB(hyp, inf, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF ), x)); 
                case 2
                    AF = @(x)(acqPI(hyp, inf, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF ), x)); 
                case 3
                    AF = @(x)(acqEI(hyp, inf, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, 1:i-1 ,iAF,run)', sPower(1:i-1, iAF ), x)); 
            end

            % We run a multi-start optimization on the acquisition function to choose the next measurement point. 
            if optimizationAlgo == 1
                [yawNext, afMax] = optimizeAcquisitionFunction(AF, xMin, xMax, nStarts);
            else
                [yawNext] = patternsearch(@(x)(-AF(x)),(xMin + 0.5*(xMax-xMin))',[],[],[],[], xMin ,xMax,[] ,options); %it seems that this code doesn't work yet
                afMax = AF(yawNext); %sYaw(:, end ,iAF,run)' 
            end
              
			% We store the selected try-out point, look up the function value and turn it into a measurement.
            sYaw(1:nYawInput, i, iAF, run) = yawNext;
            
            %BA TEST
%             maxPower = max(sPower(:, iAF, run)); 
%             powerImprovement(i, iAF, run) = sPower(i, iAF, run)-sPower(i-1, iAF, run); 
%             averageIncrease = gamma*(1/(i-1))*(maxPower - power0);
%             if powerImprovement >= averageIncrease
%                 tau(i, iAF, run) = beta*tau(i-1, iAF, run);
%             else
%                 tau(i, iAF, run) = tauInitial; 
%             end
            
            %take measurement of new point
            if realTest >= 1 %are we testing in wind tunnel? 
                fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f | %8.4f | %10.2e | %16s | %8.4f |',i, sYaw(turbine1, i, iAF, run), sYaw(turbine2, i, iAF, run), sYaw(turbine3, i, iAF, run), sLx(i,iAF, run) , sLf(i,iAF, run), sSn(i,iAF, run), 'BO')
                sPower(i, iAF, run) = setYawGetPower(arduinoSerial, sYaw(:, i, iAF, run)); 
                fprintf('%16.4f | %12.4f | %13f | \n',nlml, sPower(i, iAF, run) , toc);
            else
                sf(i, iAF, run) = OptimizationFunction(sYaw(1:nYawInput, i, iAF, run));
                sPower(i, iAF, run) = sf(i, iAF, run) + sfh*randn(1,1);
                if showProgress == 1
                    fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f | %8.4f | %10.2e | %16s | %8.4f |',i, sYaw(turbine1, i, iAF, run), sYaw(turbine2, i, iAF, run), sYaw(turbine3, i, iAF, run), sLx(i,iAF, run) , sLf(i,iAF, run), sSn(i,iAF, run), 'BO')
                    fprintf('%16.4f | %12.4f | %13f | \n',nlml, sPower(i, iAF, run) , toc);
                end
            end
            
            % We calculate the prior distribution for this new point.   
            yawM = sYaw(1:nYawInput, 1:i ,iAF,run)'; %measurements we did so far
            powerM = sPower(1:i, iAF,run); %output of measurements
            
            %We let the algorithm make a recommendation of the input, based on all data so far. This is equal to the highest mean. We use this to calculate the instantaneous regret.
            if optimizationAlgo == 1
                [yawOpt, powerOpt] = optimizeAcquisitionFunction(@(x)acqEV(hyp.gp,@infGaussLik, meanfunc, covfunc,likfunc,yawM,powerM,x),xMin , xMax, nStarts);
            else
                AF = @(x)acqEV(hyp.gp,@infGaussLik, meanfunc, covfunc,likfunc,yawM,powerM,x);
                [yawOpt] = patternsearch(@(x)(-AF(x)), sYaw(1:nYawInput,1,iAF,run)' ,[],[],[],[], xMin ,xMax, [],options);
                powerOpt = AF(yawOpt);
            end
            
            %add recommendations of maximum yaw and power to all recommendations
            sRecommendations(1:nYawInput, i, iAF, run) = yawOpt;
            sRecommendationBelievedValue(i, iAF, run) = powerOpt;
            
            if realTest < 1
                sRecommendationValue(i, iAF, run) = OptimizationFunction(yawOpt);  
            end
        end 
        
        disp('The recommended yaws are:');
        disp(sYaw(:, end, iAF, run));
        disp(['The estimated power for this yaw is: ',num2str(powerOpt)]);
        
        if  realTest == 1 
            finalMax = setYawGetPower(arduinoSerial, sRecommendations(:, end , iAF, run)  );
            disp(['The measured power for this yaw is: ', num2str(finalMax)]);
            measurementDifferencePercentage = (finalMax - powerOpt)/powerOpt*100; 
            disp(['Percentage difference between measured power and estimated max power is: ', num2str(measurementDifferencePercentage),'%']); 
            greedyPower =  sPower(1,iAF,run); 
            powerImprovementPercentage = i((finalMax - greedyPower)/greedyPower*100); 
            disp(['The power improvement for this yaw is: ',num2str(powerImprovementPercentage),'%'])
        else
            %disp(['The real power for this yaw is: ',num2str(sRecommendationValue(i, iAF, run))]);
        end
     
		% If desired, we also generate a plot of the result.
		if displayPlots ~= 0
            % We start by displaying the Gaussian process resulting from the measurements. We make the calculations for the trial points.
            %Optimizing hyperparameters for the last time and display most recent hyperparameters
            if optimizeHyperparameters == 1 
                hyp.gp = minimize(hyp.gp, @gp, -100, inf, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, : ,iAF,run)', sPower(:, iAF,run));
                disp('The optimized hyperparameters are:');
                disp(['l_x = ',num2str(exp(hyp.gp.cov(1)))]);
                disp(['l_f = ',num2str(exp(hyp.gp.cov(2)))]);
                disp(['sn = ',num2str(exp(hyp.gp.lik(1)))]);
                
                nlml = gp(hyp.gp, inf, meanfunc, covfunc , likfunc, sYaw(1:nYawInput, : ,iAF,run)', sPower(:, iAF,run) ); 
                disp(['The negative log marginal likelihood is ',num2str(nlml)]); 
                fprintf('\n');
                if plotHypOverTime == 1
                    xHyp = (1:nInputs)';
                    figure
                    grid on
                    plot(xHyp, sSn(:,iAF,run), '-', 'Color', black);
                    title('sn over time')
                    figure
                    hold on
                    grid on
                    plot(xHyp, sLx(:,iAF,run), '-', 'Color', blue);
                    title('lx over time')
                    figure
                    hold on
                    grid on
                    plot(xHyp, sLf(:,iAF,run), '-', 'Color', red);
                    title('lf over time')
                end
            end
            
            allYaw = sYaw(1:nYawInput, :,iAF,run)'; %all non zero yaw angels
            allPower = sPower(:,iAF,run); %all power measurements
            [mPost ,s2Post] = gp(hyp.gp, inf, meanfunc, covfunc, likfunc, allYaw ,allPower, Xs');
            sPost = sqrt(s2Post);
            if nYawInput ~= 1 %Do we have more then one yaw input?
                mPost = reshape(mPost, size(x1Mesh));
                sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
            end
                 
            %% We plot the resulting Gaussian process.
			if nYawInput == 1 %If we are using one yaw input
                %figNum = (run-1)*2*nAF + iAF*2 + 1;
                %figure(figNum);
                %clf(figNum);
                figure(); 
                subplot(2,1,1);
                xlabel('input');
                ylabel([afNameShort{acqSwitchedOn(1,iAF)} ' output']); %set y-label to name of corresponding acquisition function
                makeGPPlot(Xs,mPost, sPost,sYaw(1:nYawInput ,nInitialPoints+1:nInputs,iAF,run),sPower(nInitialPoints+1:nInputs,iAF,run)); %plot GP with measurements points
                plot(sYaw(1:nYawInput,1:nInitialPoints,iAF,run), sPower(1:nInitialPoints,iAF,run), 'o', 'Color', green, 'DisplayName', 'Initial Point(s)'); %plot random initialization points
                
                if realTest < 1
                    plot(Xs, fs, '-', 'Color', black, 'DisplayName', 'Optimization Function'); % We plot the true function.
                end
                legend; 
                
                subplot(2,1,2)
                hold on;
                grid on;
                xlabel('Input');
                % We check which acquisition function we are using.
               
                switch(acqSwitchedOn(1,iAF))
                    case 1 
                    AF = @(x)(acqUCB(hyp, inf, meanfunc, covfunc, likfunc,allYaw, allPower, x));
                    case 2
                    AF = @(x)(acqPI(hyp, inf, meanfunc, covfunc, likfunc,allYaw, allPower, x));
                    case 3
                    AF = @(x)(acqEI(hyp, inf, meanfunc, covfunc, likfunc,allYaw, allPower, x));    
                end
                
                % We calculate the acquisition function values at the plot points and plot those.
                afValues = zeros(1,ns);
                %tAFValues = tAF(Xs');
                for k = 1:ns
                    afValues(k) = AF(Xs(:,k));
                end
                afValues(afValues < -1e100) = min(afValues(afValues > -1e100)); % We set the insanely small numbers (which may occur when the probability is pretty much zero) to the lowest not-insanely-small number.
                plot(Xs, afValues, '-', 'Color', black);
                if plotInstant ==1 
                    drawnow; 
                end
                if realTest == 1 || saveData == 1
                    %export_fig(fullfile(folderName,'GPandACQ.png'),'-transparent');
                    fileName = strcat('GPandACQ_',num2str(afNameShort{acqSwitchedOn(1,iAF)}),'_run_',num2str(run),'.fig'); 
                    savefig(fullfile(folderName,fileName));
                end
                
                %What would be the next step?
                % yawOpt = patternsearch(@(x)(-AF(x)), initialYaw1 ,[],[],[],[],xMin,xMax);
                %0afMax = AF(yawOpt); 
                %plot(yawOpt,afMax, 'rx+');
                
                if maxDistribution == 1 %an experiment with max distribution
                    % We set up a Gaussian process to approximate the measurements, giving us the GP for our examples.
                    SPost = postCovMatrix(hyp.gp.cov, covfunc, sYaw(1:nYawInput,:,iAF,run), Xs,sfh);
                    %Particle Distribution
                    particleDistribution = particleDistr(mPost, SPost, nRounds,nParticles,ns,xMin, xMax);
                    limitDistribution = limitDistr(mPost, SPost, ns, xMin, xMax);
                    dx = (xMax - xMin)/(ns - 1); % This is the distance between two trial points.
                    maxInterval = maxInterval(limitDistribution, 0.75, dx, Xs); 
                    
                    % We plot the results.
                    figure(6);
                    clf(6);
                    hold on;
                    grid on;
                    xlabel('Input');
                    ylabel('Particle distribution');
                    particleDistributionPlot = plot(Xs, particleDistribution, '-', 'Color', 'blue');
                    
                    
                    axis([xMin,xMax,0,3.5]);
                    
                    surfaceL = dx*trapz(limitDistribution);
                    surfaceT = dx*trapz(confidenceMax); 
                    meanMaxDist = mean(limitDistribution);
                    [maxDist, imaxDist]   = max(limitDistribution);
                    xMaxDist = Xs(imaxDist);
                    x1 = xMaxDist - 0.05*(xMax-xMin);
                    x2 = xMaxDist + 0.05*(xMax-xMin);
                    confidenceMax = limitDistribution(Xs>=x1 & Xs<=x2);
                    Xs2 = linspace(x1,x2,size(confidenceMax,1)); 
                    plot(Xs2', confidenceMax, 'LineWidth', 2); 
                    plot([xMaxDist xMaxDist], [0 3.5], 'r--');
                    plot([x1 x1], [0 3.5], 'r--')
                    plot([x2 x2], [0 3.5], 'r--')
                    %                     XsMaxDistr = linspace(x1, x2, ns/10);
                    %surface1 = trapz(XsMaxDistr, limitDist);
                    %plot(xMaxDist, maxDist, 'ro');
                    % We also add the true maximum distribution and the limit distribution to the plot.
                    limitDistributionPlot = plot(Xs, limitDistribution, '-', 'Color', red, 'LineWidth', 1);
                   
                    %load('TrueMaximumDistribution');
                    %trueDistribution = plot(Xs, maxDist, '-', 'Color', blue);
                    legend([limitDistributionPlot, particleDistributionPlot], 'Limit distribution', 'Particle distribution', 'Location', 'NorthWest');
                    
                end
            else 
                if realTest ~= 1 && iAF == 1 && run == 1
                    %plot real function
                    figure
                    hold on;
                    grid on;
                    meshc(x1Mesh,x2Mesh,fsMesh);
                    surface(x1Mesh,x2Mesh,fsMesh);
                    %surf(x1Mesh, x2Mesh, fsMesh);
                    colormap('jet');
                    xlabel('x_1');
                    ylabel('x_2');
                    zlabel('Branin function output f(x_1,x_2)');
                    view([20,25]);
                    axis([xMin(1),xMax(1),xMin(2),xMax(2),-350,100])
                    if realTest == 1 || saveData == 1
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
                title([afNameShort{acqSwitchedOn(1,iAF)} ' Gaussian Process Plot, run: ', num2str(run)]); %set y-label to name of corresponding acquisition function
                
                makeGPPlot3D(x1Mesh, x2Mesh, mPost, sPost, sYaw(1:nYawInput,:,iAF, run)', sPower(:, iAF,run),plotConfPlanes); 
                %axis([xMin(1),xMax(1),xMin(2),xMax(2),-350,100]);
                axis auto
                view([20,25]); %viewpoint specification
                if realTest == 1 || saveData == 1
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
                
                switch acqSwitchedOn(1,iAF)
                    case 1
                    AF = @(x)(acqUCB(hyp, inf, meanfunc, covfunc, likfunc, sYaw(1:nYawInput, : ,iAF,run)', sPower(:,iAF,run), x));
                    case 2
                    AF = @(x)(acqPI(hyp, inf, meanfunc, covfunc, likfunc,sYaw(1:nYawInput, : ,iAF,run)',sPower(:,iAF,run), x));
                    case 3
                    AF = @(x)(acqEI(hyp, inf, meanfunc, covfunc, likfunc,sYaw(1:nYawInput, : ,iAF,run)',sPower(:,iAF,run), x));
                end
                %What would be the next point to sample? 
                [yawLast] = patternsearch(@(x)(-AF(x)),(xMin + 0.5*(xMax-xMin))',[],[],[],[], xMin ,xMax,[] ,options); %it seems that this code doesn't work yet
                afLastMax = AF(yawLast);
                
                % We calculate the acquisition function values at the plot points and plot those.
                afValues = AF(Xs');
                
                afValues(afValues < -1e100) = min(afValues(afValues > -1e100)); % We set the insanely small numbers (which may occur when the probability is pretty much zero) to the lowest not-insanely-small number.
                afValues = reshape(afValues, nsPerDimension, nsPerDimension); % We put the result in a square format again.
                
                meshc(x1Mesh,x2Mesh,afValues);
                surface(x1Mesh,x2Mesh,afValues);
                
                scatter3(yawLast(1), yawLast(2), afLastMax, 'ro', 'filled', 'DisplayName', 'Measurment Points');
                
                view([20,25]);
                colormap('default');
                title([afNameLong{acqSwitchedOn(1,iAF)}, ' AF, run: ', num2str(run)]);
                colorbar
                if plotInstant == 1
                    drawnow; 
                end
                
                %We plot the maximum distribution
                P = zeros(ns,ns);
				for i = 1:ns
					for j = 1:ns
						mut = mPost(i) - mPost(j);
						Sigmat = SPost(i,i) + SPost(j,j) - SPost(i,j) - SPost(j,i);
						P(i,j) = erf(mut/sqrt(2*Sigmat))/2 + 1/2;
					end
					P(i,i) = 1/2;
				end
				mat = diag(diag(ones(ns,ns)*P)) - P;
				outcome = zeros(ns,1);
				mat(end,:) = ones(1,ns); % We set the bottom row equal to ones.
				outcome(end) = 1; % We set the bottom element of the outcome equal to one.
				limitDist = mat\outcome; % These are the probabilities that each point is larger than any of the other points, according to the particle method.
				limitDist = limitDist/prod((xMax - xMin)/(nsPerDimension - 1)); % We turn the result into a PDF.
				limitDist = reshape(limitDist, nsPerDimension, nsPerDimension); % We put the result in a square format again.
				
				% And now we plot the true maximum distribution.
				figure
				hold on;
				grid on;
				xlabel('x_1');
				ylabel('x_2');
				zlabel('Maximum distribution');
				meshc(x1Mesh,x2Mesh,limitDist);
				surface(x1Mesh,x2Mesh,limitDist);
				if useColor == 0
					colormap(((0:0.01:0.8)'*ones(1,3)).^.8)
				else
					colormap('default');
				end
				view([20,25]);
				if exportFigs ~= 0
					export_fig(['MaximumDistribution2DRun',num2str(run),'.png'],'-transparent');
				end
                if realTest == 1 || saveData == 1
                    %export_fig(fullfile(folderName,'GPandACQ.png'),'-transparent');
                    fileName = strcat('ACQ_',num2str(afNameShort{acqSwitchedOn(1,iAF)}),'_run_',num2str(run),'.fig'); 
                    savefig(fullfile(folderName,fileName));
                end
                
                %We reset the hyperparameters to the original ones
                hyp.gp.cov= [ log(lx) log(lf)]; 
                hyp.gp.lik = log(sn);
                
            end %End of check whether how much inputs we have (2d or 3d plot)
		end % End of check whether we should make plots.
	end % End of iterating over acquisition functions.
end % End of experiment runs.
disp(['We are done with all the experiments! The time passed is ',num2str(toc),'.']);

%% We save all the data we have generated from wind tunnel test
if realTest == 1 || saveData == 1
    save(fullfile(folderName, 'Experiment_Data.mat')); %save experiment data
end
%% When doing experiments with a sample function we can calculate the error and cumulative regret

if realTest ~= 1
    % Now that we're done iterating, we calculate the instantaneous and cumulative regrets. Note that for the first we need to use the recommended points of the GPs (the highest mean) while for the
    % latter we need to use the points that were actually tried out.
    if optimizationAlgo == 1
        [xOptTrue, fOptTrue] = optimizeAcquisitionFunction(OptimizationFunction, xMin, xMax, nStarts); % Sometimes this optimization function gives the wrong optimum. When your graphs look odd, try running this block again.
    else
        [xOptTrue, fOptTrue] = optimizeAcquisitionFunction(OptimizationFunction, xMin, xMax, nStarts); % Sometimes this optimization function gives the wrong optimum. When your graphs look odd, try running this block again.
    end
    meanRecommendationValues = mean(sRecommendationValue, 3);
    meanError = fOptTrue - meanRecommendationValues;
    meanObtainedValues = mean(sf, 3); %mean over runs
    meanObtainedRegret = fOptTrue - meanObtainedValues;
    meanRegret = cumsum(meanObtainedRegret, 1);

    % We make a plot of the regret over time.
    colors = [red;blue;yellow;green;grey];
    figure;
    %clf(1);
    hold on;
    grid on;
    for i = 1:nAF
        plot(0:nInputs, [0; meanRegret(:,i)], '-', 'Color', colors(i,:));
    end
    xlabel('Measurement number');
    ylabel('Cumulative regret over time');
    legend(afNameShort(acqSwitchedOn(1,:)),'Location','NorthEast');
    % axis([0,nInputs,0,50]);
    if realTest == 1 || saveData == 1
                    export_fig(fullfile(folderName,'CumulativeRegret.png'),'-transparent');
    end
    
    % We make a plot of the error over time.
    figure;
    %clf(2);
    hold on;
    grid on;
    for i = 1:nAF
        plot(0:nInputs, [fOptTrue - mean(fs); meanError(:,i)], '-', 'Color', colors(i,:));
    end
    xlabel('Measurement number');
    ylabel('Recommendation error over time');
    legend(afNameShort(acqSwitchedOn(1,:)),'Location','NorthEast');
    % axis([0,nInputs,0,0.5]);
    if realTest == 1 || saveData == 1
        export_fig(fullfile(folderName,'RecommendationError.png'),'-transparent');
    end
    
    % We display the error in the final recommendations.
    disp(['The true power maximum is: ',num2str(fOptTrue),'The yaws are: '])
    disp(xOptTrue); 
    disp('The average final recommendation errors were:');
    disp(meanError(end,:)); 
    
    
end