function val = ac_func(m, hyp, inf, mean, cov, lik, x, y, xs)
% AC_FUNC  the acquisition function used in the PI framework for GP
%   optimization. Intuitively, we would like to choose an x that gives us 
%   values close to the target value.

[mu,s2] = gp(hyp, inf, mean, cov, lik, x, y, xs);

%xi = 0.5; 
%z = (yt-target-xi)./sqrt(s2);
%val = 
%val = (m - mu)./sqrt(s2);
kappa = 2; 
val = mu + kappa*sqrt(s2);
%kappa = 5; 
%val = yt+kappa*sqrt(s2);
epsilon = 0.1;
xi = 0.2; 
%PI = normcdf((mu - m - epsilon)./sqrt(s2));
%UCB = mu + 1*s2
