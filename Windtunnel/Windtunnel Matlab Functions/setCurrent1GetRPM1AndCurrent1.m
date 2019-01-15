 %% This function returns RPM 1 and Current1 from arduino
 function [RPM1,measuredCurrent1] = setCurrent1GetRPM1AndCurrent1(arduinoSerial, value)     %value is in mA, min 0 max 300
 
    servalue = sprintf('%d;', 5, value);        % set mode = 5 to arduino which returns RPM1 and Current1
    fprintf(arduinoSerial, servalue);                   % sending mode = 5
    pause(6);
    
    %read RPM1
    RPM1 = fscanf(arduinoSerial);                        % reading what is in the serial
    while startsWith(RPM1,'RPM') == 0        % waiting for "RPM" to appear in serial
        RPM1 = fscanf(arduinoSerial);
    end

    RPM1 = strsplit(RPM1,':');
    RPM1 = strsplit(string(RPM1(2)), ';');
    RPM1 = str2double(RPM1(1));

    %read Current1
    measuredCurrent1 = fscanf(arduinoSerial);            % reading what is in the serial
    while startsWith(measuredCurrent1,'Current') == 0        % waiting for "Current" to appear in serial
        measuredCurrent1 = fscanf(arduinoSerial);
    end
    
    measuredCurrent1 = strsplit(measuredCurrent1,':');
    measuredCurrent1 = strsplit(string(measuredCurrent1(2)), ';');
    measuredCurrent1 = str2double(measuredCurrent1(1));
    measuredCurrent1 = measuredCurrent1 / 1000000;
 end