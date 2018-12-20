function power = setYawGetPowerFLORIS(yaw, layout, controlSet, subModels)
%yaw must be column vector, if it is not, we transpose it
 if size(yaw,2) ~= 1
    yaw = yaw'; 
 end
 
yawAnglesIFArray = yaw; 
florisRunner.controlSet.yawAngleIFArray = yawAnglesIFArray;
florisRunner = floris(layout, controlSet, subModels);
florisRunner.run
f = zeros(1,florisRunner.layout.nTurbs);
for k = 1:florisRunner.layout.nTurbs
    f(k) = florisRunner.turbineResults(k).power*1e-06;
end
power = sum(f);
end