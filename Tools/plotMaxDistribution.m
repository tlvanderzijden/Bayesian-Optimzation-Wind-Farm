function maxDistributionPlot = plotMaxDistribution(maxDistribution, x, xMin, xMax, typeOfDistribution)

color = addcolours;
nInputs = size(xMin); 
grid on
if nInputs == 1 %make sure distribution and x are of the same dimensions
    if size(maxDistribution,2) > 1
        maxDistribution = maxDistribution';
    end
    
    if size(x,2) > 1
        x = x';
    end
    
    xlabel('Yaw');
    ylabel(typeOfDistribution);
    
    axis([xMin,xMax,0, max(maxDistribution)*1.1]);
    maxDistributionPlot = plot(x, maxDistribution, '-', 'Color', color.red, 'LineWidth', 1);
    legend(typeOfDistribution,'Location', 'NorthWest');
else
    x1Mesh = x(:,1:size(x,1));
    x2Mesh = x(:,size(x,1)+1:end); 
    
    grid on;
    hold on; 
    xlabel('Yaw 1');
    ylabel('Yaw 2');
    zlabel('Maximum distribution');
    meshc(x1Mesh,x2Mesh,maxDistribution);
    surface(x1Mesh,x2Mesh,maxDistribution);
    colormap('default');
    view([20,25]);

end


end
