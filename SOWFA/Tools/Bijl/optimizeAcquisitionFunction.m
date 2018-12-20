function [xOpt, yOpt] = optimizeAcquisitionFunction(af, xMin, xMax, nStarts, dimsLeft, x0)
%[xOpt, yOpt] = optimizeAcquisitionFunction(af, xMin, xMax, nStarts, dimsLeft, x0)
%optimizeAcquisitionFunction Optimizes an acquisition function.
% It takes the acquisition function as argument, together with the minimum and maximum input, and the number of starting points used in each dimension for the multi-start optimization.
if nargin < 5
	dimsLeft = length(xMin);
	x0 = (xMin + xMax)/2;
end

% If statement added by T.L. van der Zijden 23-11-2018 to orignial optimizeAcquisition function 
% When evaluating the acquisition function in fmincon there must be a scalar output, when evaluating x0. 
% So x0 must be inverted if the output of the acquisition function is a vector
% if size(af(x0),1) > 1 
%     x0 = x0'; 
% end

options = optimset('Display', 'off');
%error = size(af(x0),1); 
% We recursively walk through all dimensions, to make sure that we have nStarts^nDim starting points.
for i = 1:nStarts
	% We divide the current dimension we're looking at in blocks and pick a random starting point in each block.
	blockWidth = (xMax(dimsLeft)-xMin(dimsLeft))/nStarts;
	x0(dimsLeft) = xMin(dimsLeft) + blockWidth*(i-1/2) + blockWidth*(rand(1,1)-1/2);
	if dimsLeft == 1
		% If we are in the final dimension, we use the fmincon function.
		xFound = fmincon(@(x)(-af(x)), x0', [],[],[],[],xMin,xMax,[],options);
		yFound = af(xFound);
	else
		% If we have multiple dimensions left to go, we recursively call the optimizeAcquisitionFunction again.
		[xFound,yFound] = optimizeAcquisitionFunction(af, xMin, xMax, nStarts, dimsLeft-1, x0);
	end
	% If we are in the first iteration or have found a better point, we store it.
	if (i == 1) || (yFound > yOpt)
		xOpt = xFound;
		yOpt = yFound;
	end
end

end

