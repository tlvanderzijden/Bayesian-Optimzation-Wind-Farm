 %% this function connects to Arduino
 function connectToArduino(arduinoSerial)    
 
    % Making sure all previous connections are cleared.
    fclose(arduinoSerial);
    disp('All connections with Arduino cleared.');

    % Connecting with Arduino
    fopen(arduinoSerial);
    disp('Succesful connection with Arduino, pausing to wait for correct start position.');

    %wait for Arduino to be ready
    pause(10)
    output = fscanf(arduinoSerial);
    while startsWith(output,'Done;') == 0
        output = fscanf(arduinoSerial);
    end
    disp('End of pause.');
 end