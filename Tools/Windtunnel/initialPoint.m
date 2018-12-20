function point = initialPoint(xMin,xMax,nInitialPoints, iteration)
% This function gets the number of initial points before starting with Bayesian Optimization
% and range of the inputs, and returns the right locations to look.
%Check dimension of xMin and xMax
if size(xMin, 1) > 1
    xMin = xMin';
    xMax = xMax';
end

D = size(xMin, 2); 

if D == 1
    point = rand(D,1)*(xMax-xMin)+xMin; 
else
if nInitialPoints > 6 && iteration > 6
    point = rand(size(xMin,2),1).*(xMax-xMin)' + xMin' ;
else
    nInitialPoints = 6;
end
    
xLength = [xMax(1,1)-xMin(1,1),xMax(1,2) - xMin(1,2)];


    if nInitialPoints == 1
        iP = xMin + xLength * .5;
        point = iP(iteration,:)'; 
    elseif nInitialPoints == 2
        iP = xMin + xLength/4;
        iP(2,:) = xMin + xLength * .75;
        point = iP(iteration,:)'; 
    elseif nInitialPoints == 3
        iP = xMin .* ones(3,2) + repmat(xLength,3,1) .*[.25,.25;.5,.75;.75,.25];
        point = iP(iteration,:)'; 
    elseif nInitialPoints == 4
        iP = repmat(xMin + xLength * .25, 4, 1);
        iP = iP.*[1,1;1,-1;-1,-1;-1,1];
        point = iP(iteration,:)'; 
    elseif nInitialPoints == 5
        iP = repmat(xMin + xLength * .25, 5, 1);
        iP(3,:) = xMin + xLength * .5;
        iP = iP.*[1,1;1,-1;1,1;-1,-1;-1,1];  
        point = iP(iteration,:)'; 
    elseif nInitialPoints == 6
        iP = xMin .* ones(nInitialPoints,2) + repmat(xLength,6,1) .*[.25,.75;.5,.75;.75,.75;.25,.25;.5,.25;.75,.25];
        point = iP(iteration,:)'; 
    end
    
    
%     plot(iP(:,1),iP(:,2),'*');
%     hold on
%     axis([xMin(1,1),xMax(1,1),xMin(1,2),xMax(1,2)])
end
end
