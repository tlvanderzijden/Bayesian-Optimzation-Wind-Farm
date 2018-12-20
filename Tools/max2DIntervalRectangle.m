function [xBound ,yBound, volume] = max2DIntervalRectangle(maxDist, probabilityInterval, xMin, xMax)
%This function computes a rectangular confidence interval of the 2-input maximum distribution
%It gradually expands a probability side by side  and it computes the volume under the surface. 
%When no more improvement can be made on one side it stops expanding.
%Used as a test in MultiDimensional_Optimization, but finally not used in thesis
nsPerDimension = size(maxDist,1); 
x1Range = linspace(xMin(1),xMax(1),nsPerDimension);
x2Range = linspace(xMin(2),xMax(2),nsPerDimension);
[x1Mesh,x2Mesh] = meshgrid(x1Range,x2Range);

probabilityVolume = 0; 
[rowMax ,colMax]  = find(max(max(maxDist))==maxDist);

additionMatrix = diag([-1; 1; -1;1]); 
allZero = 0;    
xminStart = 1;
xmaxStart = 1;
yminStart = 1;
ymaxStart = 1; 
if colMax == 0, xminStart =0; end
if colMax == size(maxDist,2), xmaxStart = 0; end
if rowMax == 0, yminStart =0; end
if rowMax == size(maxDist,1), xminStart = 0; end

bounds = [colMax-xminStart; colMax+xmaxStart; rowMax-yminStart; rowMax+yminStart ]; %these are the indexes of the bounds in the mesh
boundaries = [1, size(x1Range,2); 1, size(x1Range,2); 1, size(x2Range,2); 1, size(x2Range,2)]; %boundaries for the indices
run = 2; 
%calculate probability interval for smallest volume
maxDistRangeBegin = maxDist(bounds(3):bounds(4),bounds(1):bounds(2)); 
probabilityVolume(1) = trapz(bounds(1):bounds(2),trapz(bounds(3):bounds(4),maxDistRangeBegin));%calculate probability interval for smallest mesh
if probabilityVolume(1)>1
    warning('something is wrong with the probability volume'); 
end
totalVolume = trapz(x2Range, trapz(x1Range, maxDist)); 
while probabilityVolume < probabilityInterval
    for i = 1:4
        %Check if probability volume didn't exceed probability interval
        if bounds(i)>boundaries(i,1) && bounds(i)<boundaries(i,2)%check if we do not exceed xMin and xMax
            bounds = bounds + additionMatrix(:,i);
        else
            additionMatrix(i,i) = 0;
        end
        newRangeX = x1Range(bounds(3):bounds(4));
        newRangeY = x2Range(bounds(1):bounds(2));
        maxDistRange = maxDist(bounds(3):bounds(4),bounds(1):bounds(2));
        probabilityVolume(run) = trapz(newRangeY,trapz(newRangeX,maxDistRange));
        if probabilityVolume(run)-probabilityVolume(run-1) < 0.001
            additionMatrix(i,i) = 0;
            if ~any(any(additionMatrix))
                disp('We cannot get improvement anymore. Decrease probability interval');
                allZero = true;
                break
            else
                allZero = false;
            end
        end
        run = run+1; %keep track of how many expansions we did;
    end
    if allZero, break ,end
end
totalVolume = trapz(x2Range,trapz(x1Range,maxDist)); 
xBound = [x1Mesh(1, bounds(1)); x1Mesh(1, bounds(2))] ;
yBound = [x2Mesh(bounds(3),1); x2Mesh(bounds(4,1),1)]  ;
volume = probabilityVolume(end); 
end