 %% this function connects to Arduino
 function connectToArduino(arduinoSerial)
    
    % Preparing for connection with Arduino
    % Make sure all connections with the Arduino are cleared up for a clean connection
    fclose(arduinoSerial);
    disp('All connections with Arduino cleared.');

    % Connecting with Arduino
    fopen(arduinoSerial);
    disp('Succesful connection with Arduino, pausing to wait for correct start position.');
    
    pause(10)
    %wait for Arduino to be ready
    output = fscanf(arduinoSerial);
    while startsWith(output,'Done;') == 0
        output = fscanf(arduinoSerial);
    end
    
    disp('End of pause.');
 end