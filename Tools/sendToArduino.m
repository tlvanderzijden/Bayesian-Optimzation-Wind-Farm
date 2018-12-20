function sendToArduino(arduinoSerial,mode, value)
    pause(4)
    if strcmp(mode, 'adjustWindAngle')
    mode = 4; %This mode is used to adjust wind. See arduino code windtunnel_2018 for different modes
    end
    % creating serial data to be sent
    servalue = sprintf('%d;', mode, value);
    
    % sending data
    fprintf(arduinoSerial, servalue);
    
    disp('Waiting for Megan to reach desired settings.');

    %wait for Arduino to be ready
    output = fscanf(arduinoSerial);
    while startsWith(output,'Done;') == 0
        output = fscanf(arduinoSerial);
    end
    disp('Done.');
 end