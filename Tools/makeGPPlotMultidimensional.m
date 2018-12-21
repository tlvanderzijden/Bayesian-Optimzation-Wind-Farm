function [handle1,handle2] = makeGPPlotMultidimensional(x1Mesh, x2Mesh, mPost, sPost,xm,fm, plotConfPlane, typeOfPlot)
%makeGPPlot Will generate a plot in the given figure frame for the given GP output. It can be used to make a surface plot or a contour
% The makeGPPlot function makes a plot of the results of a GP regression algorithm. Required parameters are the following.
%	xp: The input points for the plot.
%	mup: The mean values at the plot points.
%	stdp: The standard deviation at the plot points.
%   xm: matrix with measurement points (nxd)
%   fm: output of measurements points
%   plotConfPlane: display confidence interval planes? (on/off)(string)
%   typeOfPlot: contour/surface (string)
% The xp, mup and stdp should all be vectors of the same size.
% As output, there will be two plot handles.
%	handle1: The handle of the mean line which is plotted.
%	handle2: The handle of the grey area representing the 95% uncertainty region.
% Either of these handles can be used for making a proper figure legend later on, if necessary.
white = [1 1 1];
blue = [0 0 0.8];
grey = [0.8 0.8 1];
red = [0.8 0 0];

% We make a plot of the result.
hold on;
grid on;
axis auto

if strcmp(typeOfPlot, 'surface')
sMid = surface(x1Mesh, x2Mesh, mPost);
%set(sMid,'FaceAlpha',0.8, 'FaceColor',[230/255,230/255,230/255]);

if (strcmp(plotConfPlane,'on'))
    sDown = surface(x1Mesh, x2Mesh, mPost - 2*sPost); %std plane
    set(sDown,'FaceAlpha',0.5, 'LineStyle', 'none', 'FaceColor',[0, 0,1]); %transparency and linestyle
    sUp = surface(x1Mesh, x2Mesh, mPost + 2*sPost);
    set(sUp,'FaceAlpha',0.5, 'LineStyle', 'none', 'FaceColor',[0,0,1]); %transparency            
end

axis auto
%axis([min(xp), max(xp), 1.2*min(mup-2*stdp), 1.2*max(mup+2*stdp)]); % We set the plot bounds to the correct value.
colorbar
if nargin > 5
    scatter3(xm(:,1), xm(:,2), fm(:,1), 'ro', 'filled', 'DisplayName', 'Measurment Points'); %plot measurement points
end
view([20,25]); %viewpoint specification
else % plot contour
    subplot(1,2,1) %plot mean
    contourf(x1Mesh, x2Mesh,mPost);  
    hold on
    %s = scatter(xm(:,1), xm(:,2), 'filled'); 
    scatter(xm(:,1), xm(:,2),100, 'MarkerEdgeColor',[0 0 0],...
              'MarkerFaceColor',[1 1 1],...
              'LineWidth',1.5); 
    %set(s, 'LinedWidth', 1, 'FaceColor', [ 1 1 1] , 'MarkerEdgeColor', [ 0 0 0 ]); 
    axis([x1Mesh(1,1), x1Mesh(1,end), x2Mesh(1,1), x2Mesh(end,1)]); 
    colorbar
    subplot(1,2,2) % plot variance
    contourf(x1Mesh, x2Mesh,sPost);  
    hold on; 
    scatter(xm(:,1), xm(:,2),100, 'MarkerEdgeColor',[0 0 0],...
              'MarkerFaceColor',[1 1 1],...
              'LineWidth',1.5); 
%    colormap default; 
    colorbar
     axis([x1Mesh(1,1), x1Mesh(1,end), x2Mesh(1,1), x2Mesh(end,1)]); 
end
end

