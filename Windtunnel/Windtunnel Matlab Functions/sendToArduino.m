function sendToArduino(arduinoSerial,mode, value)
    %waiting for arduino to be ready
    pause(4)

% creating serial data to be sent
    servalue = sprintf('%d;', mode, value);
    
    % sending data
    fprintf(arduinoSerial, servalue);
    
    disp('Waiting for Megan to reach desired settings.');
    %wait for Arduino to be ready
    output = fscanf(arduinoSerial);
    if mode ~= 5
        while startsWith(output,'Done;') == 0
            output = fscanf(arduinoSerial);
        end
    end
    disp('Done.');
 end