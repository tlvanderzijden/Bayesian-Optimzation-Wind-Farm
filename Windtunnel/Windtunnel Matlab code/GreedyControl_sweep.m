close all
clear all
clc

%% Preparing for connection with Arduino
% Make sure the baud rate and COM port is same as in Arduino
% Make sure all connections with the Arduino are cleared up for a clean connection
instrreset
arduinoSerial = serial('COM7','BAUD', 9600);
connectToArduino(arduinoSerial)

%% Setting up variables
windAngle = -12;
windStep = 3;
windSteps = 9;

k = 1;
data = zeros(windSteps,5);
KT=3.52E-3; %Nm A^-1

%% Setting windtunnel initial conditions
%test connection
disp('Starting test measurement.');
[RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents(arduinoSerial, 0, 0, 0)

disp('Please check the connections. Press any key to continue.');
pause

%%
tic
for i = 1:windSteps
    sendToArduino(arduinoSerial, 4, windAngle)
    [RPMs,measuredCurrents] = setYawsGetRPMsAndCurrents(arduinoSerial, 0, 0, 0);
            Power1 = measuredCurrents(1) * KT * RPMs(1) *pi/30;
            Power2 = measuredCurrents(2) * KT * RPMs(2) *pi/30;
            Power3 = measuredCurrents(3) * KT * RPMs(3) *pi/30;
            totalPower = Power1 + Power2 + Power3;

            %save data
            data(i,1) = windAngle;
            data(i,2) = Power1;
            data(i,3) = Power2;
            data(i,4) = Power3;
            data(i,5) = totalPower;

            %display progress
            disp(['Measured total power is: ',num2str(totalPower),' W. Time passed is: ',num2str(toc), ' seconds.']);
    windAngle = windAngle + windStep;
end

disp('Finished!');
toc
