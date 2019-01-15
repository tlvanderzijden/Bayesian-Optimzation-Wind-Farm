close all
clear all
clc

%% Preparing for connection with Arduino
% Make sure the baud rate and COM port is same as in Arduino
% Make sure all connections with the Arduino are cleared up for a clean connection
instrreset
arduinoSerial = serial('COM7','BAUD', 9600);
connectToArduino(arduinoSerial)

%%
windAngle=0;
KT=3.52E-3; %Nm A^-1
sendToArduino(arduinoSerial,4,windAngle);

%%
clc
[RPMs,measuredCurrents] = setYawsGetRPMsAndCurrents_instant(arduinoSerial, 0, 0, 0)
Power1 = measuredCurrents(1) * KT * RPMs(1) *pi/30
Power2 = measuredCurrents(2) * KT * RPMs(2) *pi/30
Power3 = measuredCurrents(3) * KT * RPMs(3) *pi/30
totalPower = Power1 + Power2 + Power3