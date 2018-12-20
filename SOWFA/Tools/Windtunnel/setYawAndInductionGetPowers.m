 %% this function sets yaws and gets powers
 function [totalPower,Power1,Power2,Power3] = setYawAndInductionGetPowers(arduinoSerial,yaw)     %value is in mA, min 0 max 400
 %sets arduino communication variables

 %com7 is the usb port connected to the arduino
 %9600 is the baud rate set on the arduino
 %serial = serial('COM7','BAUD', 9600);

 pause(4)

 %yaw must be column vector, if it is not, we transpose it
 if size(yaw,2) ~= 1
    yaw = yaw'; 
 end
 
 yaw1 = yaw(1);
 yaw2 = yaw(2);
 yaw3 = yaw(3);
 induction1 = yaw(4);
 induction2 = yaw(5);
 induction3 = yaw(6);
 
 % set mode = 9 to arduino which sets yaws and returns RPMs and currents
 servalue = sprintf('%d;', 9, yaw1, yaw2, yaw3, induction1, induction2, induction3);        
 fprintf(arduinoSerial, servalue);                   
    
 %waiting for the wind to settle
 pause(8);
    
    %Get RPMs
    RPMs = fscanf(arduinoSerial);            % reading what is in the serial
    while startsWith(RPMs,'RPM') == 0        % waiting for "RPM" to appear in serial
        RPMs = fscanf(arduinoSerial);
    end

    % splitting returned serial data to reconstruct RPMs
    RPMs = strsplit(RPMs,':');                                            
    RPMs = strsplit(string(RPMs(2)),';');
    RPMs = str2double(RPMs(1:3));

    %Get currents
   measuredCurrents = fscanf(arduinoSerial);                        % reading what is in the serial
    while startsWith(measuredCurrents,'Current') == 0        % waiting for "Current" to appear in serial
        measuredCurrents = fscanf(arduinoSerial);
    end

    % splitting returned serial data to reconstruct measured Currents
    measuredCurrents = strsplit(measuredCurrents,':');                                            
    measuredCurrents = strsplit(string(measuredCurrents(2)),';');
    measuredCurrents = str2double(measuredCurrents(1:3));
    measuredCurrents = measuredCurrents ./ 1000000;
    
    KT=3.52E-3; %Nm A^-1
    Power1 = measuredCurrents(1) * KT * RPMs(1);
    Power2 = measuredCurrents(2) * KT * RPMs(2);
    Power3 = measuredCurrents(3) * KT * RPMs(3);
    totalPower = Power1 + Power2 + Power3;
 end