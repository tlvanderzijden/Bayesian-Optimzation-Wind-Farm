function XsInterval = maxInterval(maxDist, probabilityInterval, xMin, xMax , Xs)
k = 0;
%limitDistscaled = 100*maxDist/trapz(maxDist);
probabilitySurface = 0; 
nsPerDimension = size(maxDist,1); 
dx = prod((xMax - xMin)/(nsPerDimension - 1)); 
while probabilitySurface < probabilityInterval
    k = k + 1e-3;
    limitDistInterval = maxDist((Xs > -k) & (Xs < k));
    probabilitySurface = dx*trapz(limitDistInterval);

end

XsInterval = Xs((Xs > -k) & (Xs < k));

end