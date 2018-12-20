function [handle1,handle2] = makeGPPlot(xp, mup, stdp,xm,fm)
%makeGPPlot Will generate a plot in the given figure frame for the given GP output.
% The makeGPPlot function makes a plot of the results of a GP regression algorithm. Required parameters are the following.
%	xp: The input points for the plot.
%	mup: The mean values at the plot points.
%	stdp: The standard deviation at the plot points.
%   xm: measurement points
%   fm: output of measurements points
% The xp, mup and stdp should all be vectors of the same size.
% As output, there will be two plot handles.
%	handle1: The handle of the mean line which is plotted.
%	handle2: The handle of the grey area representing the 95% uncertainty region.
% Either of these handles can be used for making a proper figure legend later on, if necessary.
white = [1 1 1];
blue = [0 0 0.8];
grey = [0.8 0.8 1];
red = [0.8 0 0];

% We want all given vectors to be row vectors. If they're not, we flip (transpose) them.
if size(xp,2) == 1
	xp = xp';
end
if size(mup,2) == 1
	mup = mup';
end
if size(stdp,2) == 1
	stdp = stdp';
end

% We make a plot of the result.
hold on;
grid on;
handle2 = patch([xp, fliplr(xp)],[mup-2*stdp, fliplr(mup+2*stdp)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none', 'DisplayName', '95% confidence interval'); % This is the grey area in the plot.
patch([xp, fliplr(xp)],[mup-stdp, fliplr(mup+stdp)], 1, 'FaceColor', grey, 'EdgeColor', 'none', 'DisplayName','65% confidence interval'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
handle1 = plot(xp, mup, 'k-', 'LineWidth', 1, 'Color',blue, 'DisplayName', 'Mean'); % We plot the mean line.
axis([min(xp), max(xp), 1.2*min(mup-2*stdp), 1.2*max(mup+2*stdp)]); % We set the plot bounds to the correct value.

if nargin > 3
    plot(xm, fm, 'o', 'Color', red, 'DisplayName', 'Measurement Points'); % We plot ttrueDistributionhe measurement points.
end

end

