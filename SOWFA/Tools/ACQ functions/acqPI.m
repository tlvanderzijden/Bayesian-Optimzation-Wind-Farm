function val=acqPI(hyp, fOpt, inf,mean,cov,lik, xm,ym,xs)
%this file creates the Probability of Improvement acquisition function

[mu,s2] = gp(hyp.gp, inf, mean, cov, lik, xm, ym, xs);
% fsh = max(mu); % This is \hat{f}^*.
xi = hyp.acq;
z0 = (mu - fOpt-xi)./sqrt(s2);
Phi = normcdf(z0); 
%z = ((mu - fOpt - xi)./sqrt(s2));
%Phi = 1/2 + 1/2.*erf(z./sqrt(2));

%%We set up the right output value.
res = Phi;
%val = log(Phi); 
for i = 1:size(res,1)
    if res(i) > 0
        val(i,1) = log(Phi(i)); % Usually the output is Phi, but we take the logarithm because otherwise the values are just too small.
    else
        val(i,1) = -1e200; % Sometimes y becomes zero for numerical reasons. This basically means that it's extremely small. Still, it'll crash the algorithm. So we just set a default very small value here.
    end
end
end



