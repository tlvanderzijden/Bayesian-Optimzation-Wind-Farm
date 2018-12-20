function [y] = acqEIBijl(hyp,xi,fOpt,mean, cov, lik, x, y, xs)
%AFPI The acquisition function using the expected improvement.

	% We make a GP prediction for the test point.
	[mu,s2] = gp(hyp, @infGaussLik, mean,cov,lik,x,y,xs);
	mu = mu';
    s2 = s2'; 
    %fsh = max(mu); 
	% We calculate the probability of improvement. We do this using the error function.
	z = ((mu - fOpt - xi)/sqrt(s2));
	Phi = 1/2 + 1/2*erf(z/sqrt(2));
	phi = 1/sqrt(det(2*pi))*exp(-1/2*z^2);
	
	% We set up the right output value.
	res = sqrt(s2)*(z*Phi + phi);
	if res > 0
		y = log(res); % Usually the output is the equation within the logarithm, but we take the logarithm because otherwise the result is too small.
	else
		y = -1e200; % Sometimes y becomes zero for numerical reasons. This basically means that it's extremely small. Still, it'll crash the algorithm. So we just set a default very small value here.
	end
end