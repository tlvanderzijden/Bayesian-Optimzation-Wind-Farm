close all
clear all
clc

%% Preparing for connection with Arduino
% Make sure the baud rate and COM port is same as in Arduino
% Make sure all connections with the Arduino are cleared up for a clean connection

instrreset
arduinoSerial = serial('COM7','BAUD', 9600);
connectToArduino(arduinoSerial)

 %% Testing connection with Arduino
disp('Starting test measurement.');
[RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents_instant(arduinoSerial, 0, 0, 0)
disp('Please check the connections. Press any key to continue.');
pause
    
%% Setting up variables
iterations = 100;
data = zeros(iterations,5);
KT=3.52E-3; %Nm A^-1

%%
tic
for i = 1:iterations
    %display progress
    disp(['Starting test experiment run ',num2str(i),'/',num2str(iterations),'. Time passed is ',num2str(toc),' seconds.']);
    
    %measure RPMs and Currents and compute power
    [RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents_instant(arduinoSerial,0,0,0);
    Power1 = measuredCurrents(1) * KT * RPMs(1) *pi/30;
    Power2 = measuredCurrents(2) * KT * RPMs(2) *pi/30;
    Power3 = measuredCurrents(3) * KT * RPMs(3) *pi/30;
    totalPower = Power1+Power2+Power3;
    
    %save data
    data(i,1)=i;
    data(i,2)=totalPower;
    
    data(i,3)= Power1;
    data(i,4)= Power2;
    data(i,5)= Power3;
    
    data(i,6) = RPMs(1);
    data(i,7) =  measuredCurrents(1);
end
toc
disp('Finished!');

%% Determine standard deviation and variance
avg = mean(data(:,3))
sdev = std(data(:,3))

%% Plot all powers
hold on
plot(data(:,1),data(:,3), 'g')


legend('Total Power', 'Power 1', 'Power 2', 'Power 3')
axis([0 iterations 0 max(data(:,2))*1.1])

plot(data(:,1),data(:,6), 'r')
plot(data(:,1),data(:,7), 'c')