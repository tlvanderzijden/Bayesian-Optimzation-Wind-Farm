%this file creates the Expected Improvement acquisition function
function val=acqEI(hyp, fOpt, inf, mean, cov, lik, xm, ym, xs)

[mu,s2] = gp(hyp.gp, inf, mean, cov, lik, xm, ym, xs); 

xi = hyp.acq;

z0 = (mu - fOpt-xi)./sqrt(s2);
Phi = (mu- fOpt).*normcdf(z0) + sqrt(s2).*normpdf(z0);
res = Phi;
%val = log(Phi); 
for i = 1:size(res,1)
    if res(i) > 0
        val(i,1) = log(Phi(i)); % Usually the output is Phi, but we take the logarithm because otherwise the values are just too small.
    else
        val(i,1) = -1e200; % Sometimes y becomes zero for numerical reasons. This basically means that it's extremely small. Still, it'll crash the algorithm. So we just set a default very small value here.
    end
end
