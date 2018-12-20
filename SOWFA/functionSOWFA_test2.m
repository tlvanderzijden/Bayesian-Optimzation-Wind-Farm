function [varBO] = functionSOWFA(varBO)
varExperiment = SOWFA_two_dimensional_test2_19_12;
xMin = varExperiment.xMin;
xMax = varExperiment.xMax;
nYawInput = 2; % This is the dimension of the input vector.r i = 3:401
nTurbines = 3; %Number of windturbines that can be controlled
nInitialPoints = 4; 
nInputs = 50;
nStarts = 3;
% We define settings for the script.
if any(strcmp(fieldnames(varBO),'iteration'))
    varBO.iteration = varBO.iteration + 1;
else
    varBO.iteration = 1; % First iteration
end

% Copy settings from struct()
iteration = varBO.iteration;
nYawInput = varBO.nYawInput;
optimizeHyperparameters = 1; %Do we want to tune hyperparameters
startOptimizeHyp = 6; %start optimization of hyperparameters after startOptimizeHyp measurements
options = optimset('Display','off'); 

%% Hyperparameters
if iteration == 1

   
    varBO.sYaw = zeros(nTurbines, nInputs); % These are the chosen measurement points.
    varBO.sPower = zeros(nInputs,1); % These are the measured function values (with measurement noise).
    varBO.sLx = zeros(nInputs,2);
    varBO.sLf = zeros(nInputs,1);
    varBO.sSn = zeros(nInputs,1);
    varBO.sRecommendations = zeros(nTurbines, nInputs); % These are the recommendations of the optimal yaws made at the end of all the measurements.
    varBO.sRecommendationBelievedValue = zeros(nInputs,1); %These are the powers which the GP beliefs we would get at the recommended yaw angels.
    varBO.boundaries.xMin = xMin;
    varBO.boundaries.xMax = xMax; 
    
    varBO.sLx(1,:) = varExperiment.lx(1:2); 
    varBO.sLf(1) = varExperiment.lf;
    varBO.sSn(1) = varExperiment.sn;
    
    hyp.gp.cov(1:2) =  log(varBO.sLx(iteration,:));
    hyp.gp.cov(3) = log(varBO.sLf(iteration));
    hyp.gp.lik = log(varBO.sSn(iteration));
else
    %We set up the hyperparameters
    hyp.gp.cov(1:2) =  log(varBO.sLx(iteration-1,:));
    hyp.gp.cov(3) = log(varBO.sLf(iteration-1));
    hyp.gp.lik = log(varBO.sSn(iteration-1));
end

%We set up other acquisition function settings. These have been tuned manually.
hypacq = varExperiment.hyp.Acq.EI;  
hyp.acq= hypacq;

%% Kernel, mean, likelihood
covfunc = varExperiment.propGP.covfunc; 
meanfunc = varExperiment.propGP.meanfunc;
likfunc = @likGauss;
acqfunc = varExperiment.propGP.acqfunc; 

mu = 3; s2 = 0.5^2;
pg = {@priorGauss,mu,s2};
prior.cov= {pg;  pg; pg};
inf = {@infPrior,@infGaussLik,prior};

varBO.propGP.covfunc = covfunc;
varBO.propGP.meanfunc = meanfunc;
varBO.propGP.likfunc = likfunc;
varBO.propGP.inf = @infGaussLik; 


%% We start doing the experiment runs.
% We keep track of how the Bayesian Optimization

%We select first measurement points. This is the same for all acquisition functions to reduce random effects. (If one AF gets a lucky first point and another a crappy one, it's kind of unfair.)
if iteration <= nInitialPoints
    if iteration == 1
        varBO.sYaw(:,iteration) = 0;
    else
        
        varBO.sYaw(1:nYawInput, iteration) = initialPoint(xMin, xMax, nInitialPoints, iteration-1);
        %Optimizing hyperparameters
        sYaw = varBO.sYaw(1:nYawInput ,1:iteration-1)'; 
        sPower = varBO.sPower(1:iteration-1); 

        % We let the algorithm make a recommendation of the input, based on all data so far. This is equal to the highest mean.
        AFev = @(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc, sYaw,sPower,x);
        [yawOpt] = patternsearch(@(x)(-AFev(x)), (xMin + 0.5*(xMax-xMin))' ,[],[],[],[], xMin ,xMax, [],options);
        powerOpt = AFev(yawOpt);
        varBO.sRecommendations(1:nYawInput,iteration-1) = yawOpt;
        varBO.sRecommendationsValue(iteration-1) = powerOpt;
        fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f |%8.2f | %8.4f  | %12.4f |  \n',iteration, varBO.sYaw(1, iteration), varBO.sYaw(2, iteration), varBO.sYaw(3, iteration), exp(hyp.gp.cov(1)),exp(hyp.gp.cov(2)), exp(hyp.gp.cov(3)), exp(hyp.gp.lik))
    end
    
    %save hyperparameters
    varBO.sLx(iteration,:) = exp(hyp.gp.cov(1:2))'; 
    varBO.sLf(iteration) = exp(hyp.gp.cov(3)); 
    varBO.sSn(iteration) = exp(hyp.gp.lik); 
else
    %Optimizing hyperparameters
    sYaw = varBO.sYaw(1:nYawInput ,1:iteration-1)'; 
    sPower = varBO.sPower(1:iteration-1); 
    
    % We let the algorithm make a recommendation of the input, based on all data so far. This is equal to the highest mean.
    AFev = @(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc, sYaw,sPower,x);
    [yawOpt] = patternsearch(@(x)(-AFev(x)), (xMin + 0.5*(xMax-xMin))' ,[],[],[],[], xMin ,xMax, [],options);
    powerOpt = AFev(yawOpt);
    varBO.sRecommendations(1:nYawInput,iteration-1) = yawOpt;
    varBO.sRecommendationsValue(iteration-1) = powerOpt;
    
    if optimizeHyperparameters == 1 && iteration > startOptimizeHyp
        hyp.gp = minimize(hyp.gp, @gp, -100, inf, meanfunc, covfunc, likfunc, sYaw, sPower);
        %nlml = gp(hyp.gp, inf, meanfunc, covfunc , likfunc, sYaw', sPower);
    end
    
    % We run a multi-start optimization on the acquisition function to choose the next measurement point.
    
    if acqfunc == 1
        AF = @(x)(acqUCB(hyp, inf, meanfunc, covfunc, likfunc, sYaw, sPower, x));
    else
        [~, powerOpt] = optimizeAcquisitionFunction(@(x)acqEV(hyp,@infGaussLik, meanfunc, covfunc,likfunc,sYaw,sPower,x),xMin , xMax, nStarts); 
        AF = @(x)(acqEI(hyp, powerOpt, inf, meanfunc, covfunc, likfunc, sYaw, sPower, x));
    end
    %varBO.sYaw(1:nYawInput,iteration) = patternsearch(@(x)(-AF(x)),(xMin + 0.5*(xMax-xMin))',[],[],[],[], xMin ,xMax,[] ,options); 
    varBO.sYaw(1:nYawInput,iteration) = optimizeAcquisitionFunction(AF, xMin, xMax, nStarts);
    
     fprintf('%10i | %8.2f | %8.2f |%8.2f | %8.2f |%8.2f | %8.4f  | %12.4f |  \n',iteration, varBO.sYaw(1, iteration), varBO.sYaw(2, iteration), varBO.sYaw(3, iteration), exp(hyp.gp.cov(1)),exp(hyp.gp.cov(2)), exp(hyp.gp.cov(3)), exp(hyp.gp.lik))
     %save hyperparameter history
    varBO.sLx(iteration,:) = exp(hyp.gp.cov(1:2));
    varBO.sLf(iteration) = exp(hyp.gp.cov(3));
    varBO.sSn(iteration) = exp(hyp.gp.lik);
end

        