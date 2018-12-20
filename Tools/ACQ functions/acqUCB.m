function val = acqUCB(hyp,inf, mean, cov, lik, x, y, xs)
%acqUCB(hyp,kappa, inf, meanfunc, covfunc, likfunc, x, y, xs)
%this file creates the Upper confidence bound acquisition function, it
%recquires an extra kappa input that scales the variance
if nargin ~= 8 
  disp('Usage: [val] = acqUCB(hyp,kappa, inf, mean, cov, lik, x, y, xs)')
  return
end
kappa = hyp.acq; 
[mu,s2] = gp(hyp.gp, inf, mean, cov, lik, x, y, xs); 
val = (mu + kappa*sqrt(s2));
