function val = acqEV(hyp, inf, mean, cov, lik, x, y, xs)
%acqEV(hyp, inf, meanfunc, covfunc, likfunc, x, y, xs)
%this file creates the EV acquisition function

[mu,~] = gp(hyp.gp, inf, mean, cov, lik, x, y, xs); 
val = mu;

