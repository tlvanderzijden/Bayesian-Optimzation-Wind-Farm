 %% this function returns RPM 1 and current 1 from arduino
 function [RPMs,measuredCurrents] = setYawsGetRPMsAndCurrents_Instant(arduinoSerial, value1, value2, value3)     %value is in mA, min 0 max 400
    
 % set mode = 7 to arduino which sets yaws and returns RPMs and currents
 servalue = sprintf('%d;', 7, value1, value2, value3);        
 fprintf(arduinoSerial, servalue);                   
        
    %get RPMs
    RPMs = fscanf(arduinoSerial);                        % reading what is in the serial
    while startsWith(RPMs,'RPM') == 0        % waiting for "RPM" to appear in serial
        RPMs = fscanf(arduinoSerial);
    end

    % splittng returned serial data to reconstruct RPMs
    RPMs = strsplit(RPMs,':');                                            
    RPMs = strsplit(string(RPMs(2)),';');
    RPMs = str2double(RPMs(1:3));

   %get measured currents
   measuredCurrents = fscanf(arduinoSerial);                        % reading what is in the serial
    while startsWith(measuredCurrents,'Current') == 0        % waiting for "Current" to appear in serial
        measuredCurrents = fscanf(arduinoSerial);
    end

    % splittng returned serial data to reconstruct measured Currents
    measuredCurrents = strsplit(measuredCurrents,':');                                            
    measuredCurrents = strsplit(string(measuredCurrents(2)),';');
    measuredCurrents = str2double(measuredCurrents(1:3));
    measuredCurrents = measuredCurrents ./ 1000000;
    
 end