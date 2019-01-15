 %% this function returns RPM 1 and current 1 from arduino
 function [totalPower] = setYawsGetPower(arduino, value1, value2, value3)     %value is in mA, min 0 max 400
    
 %sets arduino communication variables
 %com7 is the usb port connected to the arduino
 %9600 is the baud rate set on the arduino
 %serial = serial('COM7','BAUD', 9600);
 
 % set mode = 6 to arduino which sets yaws and returns RPMs and currents
 servalue = sprintf('%d;', 6, value1, value2, value3);        
 fprintf(arduino, servalue);                   
    
 %waiting for the wind to settle
 pause(8);
    
    RPMs = fscanf(arduino);                        % reading what is in the serial
    while startsWith(RPMs,'RPM') == 0        % waiting for "RPM" to appear in serial
        RPMs = fscanf(arduino);
    end

    % splittng returned serial data to reconstruct RPMs
    RPMs = strsplit(RPMs,':');                                            
    RPMs = strsplit(string(RPMs(2)),';');
    RPMs = str2double(RPMs(1:3));

   measuredCurrents = fscanf(arduino);                        % reading what is in the serial
    while startsWith(measuredCurrents,'Current') == 0        % waiting for "Current" to appear in serial
        measuredCurrents = fscanf(arduino);
    end

    % splittng returned serial data to reconstruct measured Currents
    measuredCurrents = strsplit(measuredCurrents,':');                                            
    measuredCurrents = strsplit(string(measuredCurrents(2)),';');
    measuredCurrents = str2double(measuredCurrents(1:3));
    measuredCurrents = measuredCurrents ./ 1000000;
    
    KT=3.52E-3; %Nm A^-1
    Power1 = measuredCurrents(1) * KT * RPMs(1);
    Power2 = measuredCurrents(2) * KT * RPMs(2);
    Power3 = measuredCurrents(3) * KT * RPMs(3);
    totalPower = Power1 + Power2 + Power3;
        
    %wait for Arduino to be ready
    output = fscanf(arduinoSerial);
    while startsWith(output,'Done;') == 0
        output = fscanf(arduinoSerial);
    end
 end