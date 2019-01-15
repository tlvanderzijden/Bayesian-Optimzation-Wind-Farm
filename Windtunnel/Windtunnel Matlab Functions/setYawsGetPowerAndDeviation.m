 %% this function returns Total Power from Arduino, averages over 1000 measurements and gives standard deviation
 function [totalPower,Power1, Power2, Power3, standardDeviation] = setYawsGetPowerAndDeviation(arduinoSerial, value1, value2, value3)
    
    % setting temporary variables
    measurements = 100;
    powers = zeros(measurements,4);
    KT=3.52E-3; %Nm A^-1
    
    % wait for the wind to settle
    pause(8)
    
    % get measurements
    for i = 1 : measurements
        %measure RPMs and Currents and compute power
        [RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents_instant(arduinoSerial,value1,value2,value3);
        Power1 = measuredCurrents(1) * KT * RPMs(1);
        Power2 = measuredCurrents(2) * KT * RPMs(2);
        Power3 = measuredCurrents(3) * KT * RPMs(3);
        totalPower = Power1 + Power2 + Power3;
        
        %save data
        powers(i,2)=totalPower;
        powers(i,3)=Power1;
        powers(i,4)=Power2;
        powers(i,5)=Power3;
    end
    
    totalPower = mean(powers(:,2));
    Power1 = mean(powers(:,3));
    Power2 = mean(powers(:,4));
    Power3 = mean(powers(:,5));
    
    % Determine standard deviation and variance
    standardDeviation = std(powers(:,2));
 end