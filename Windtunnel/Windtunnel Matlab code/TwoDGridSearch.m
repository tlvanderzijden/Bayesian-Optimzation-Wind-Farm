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
windAngle = 12;
yawstep = 5;

yaw1start = -45;
yaw1end = 45;

yaw2start = -45;
yaw2end = 45;

k = 1;
iterations1 = floor((yaw1end - yaw1start) / yawstep) + 1;
iterations2 = floor((yaw2end - yaw2start) / yawstep) + 1;
data = zeros(iterations1*iterations2, 5);
KT=3.52E-3; %Nm A^-1

%% Setting windtunnel initial conditions
%test connection
disp('Starting test measurement.');
[RPMs, measuredCurrents] = setYawsGetRPMsAndCurrents(arduinoSerial, 0, 0, 0);
Power1 = measuredCurrents(1) * KT * RPMs(1) *pi/30

disp('Please check the connections. Press any key to continue.');
pause

%Setting wind angle
sendToArduino(arduinoSerial, 4, windAngle)

%%
tic
for i = 1:iterations1
    for j = 1:iterations2  
        
        %display progress
        disp(['Starting test experiment run ',num2str(j + (i-1)*iterations1),'/',num2str(iterations1*iterations2),'. Time passed is ',num2str(toc),' seconds.']);
        
        %measure RPMs and Currents and compute power
        [RPMs,measuredCurrents] = setYawsGetRPMsAndCurrents(arduinoSerial, yaw1start+yawstep*(i-1), yaw2start+yawstep*(j-1), 0);
        Power1 = measuredCurrents(1) * KT * RPMs(1) *pi/30;
        Power2 = measuredCurrents(2) * KT * RPMs(2) *pi/30;
        Power3 = measuredCurrents(3) * KT * RPMs(3) *pi/30;
        totalPower = Power1 + Power2 + Power3;

        %save data
        data(k,1) = (i-1)*yawstep+yaw1start;
        data(k,2) = (j-1)*yawstep+yaw2start;
        data(k,3) = Power1;
        data(k,4) = Power2;
        data(k,5) = Power3;
        data(k,6) = totalPower;
        k = k+1;
 
        %display progress
        disp(['Measured total power is: ',num2str(totalPower),' W. Time passed is: ',num2str(toc), ' seconds.']);
    end
end
disp('Finished!');

%% Calculate improvement
startpower = data( ceil((iterations1*iterations2)/2),6);
maxpower = max(data(:,6));
gain = ((maxpower - startpower) / startpower) * 100;
disp(['Potential power increase by yaw misalignment is: ',num2str(gain),' %.']);

%% Make a mesh of and plot total power
xt = data(:,1);
yt = data(:,2);
zt = data(:,6);

xlin = linspace(-45,45,4*sqrt(length(data(:,1))));
ylin = linspace(-45,45,4*sqrt(length(data(:,1))));

[Xt,Yt] = meshgrid(xlin,ylin);

f = scatteredInterpolant(xt,yt,zt);

Zt = f(Xt,Yt);

%figure
mesh(Xt,Yt,Zt) %interpolated
axis([-45 45 -45 45 0.00 0.5])
hold on

plot3(xt,yt,zt,'.','MarkerSize',15) %nonuniform
xlabel('Yaw 1 [degrees]')
ylabel('Yaw 2 [degrees]')
zlabel('Power [W]')


%% Make a mesh of and plot power1
x2 = data(:,1);
y2 = data(:,2);
z2 = data(:,3);

xlin = linspace(-45,45,sqrt(length(data(:,1))));
ylin = linspace(-45,45,sqrt(length(data(:,1))));

[X2,Y2] = meshgrid(xlin,ylin);

f = scatteredInterpolant(x2,y2,z2);

Z1 = f(X2,Y2);

mesh(X2,Y2,Z1) %interpolated

plot3(x2,y2,z2,'g.','MarkerSize',15) %nonuniform
xlabel('Yaw 1 [degrees]')
ylabel('Yaw 2 [degrees]')
zlabel('Power [W]')

%% Make a mesh of and plot power 2
x2 = data(:,1);
y2 = data(:,2);
z2 = data(:,4);

xlin = linspace(-45,45,sqrt(length(data(:,1))));
ylin = linspace(-45,45,sqrt(length(data(:,1))));

[X2,Y2] = meshgrid(xlin,ylin);

f = scatteredInterpolant(x2,y2,z2);

Z3 = f(X2,Y2);

mesh(X2,Y2,Z3) %interpolated

plot3(x2,y2,z2,'.c','MarkerSize',15) %nonuniform
xlabel('Yaw 1 [degrees]')
ylabel('Yaw 2 [degrees]')
zlabel('Power [W]')

%% Make a mesh of and plot power 3
x3 = data(:,1);
y3 = data(:,2);
z3 = data(:,5);

xlin = linspace(-45,45,sqrt(length(data(:,1))));
ylin = linspace(-45,45,sqrt(length(data(:,1))));

[X3,Y3] = meshgrid(xlin,ylin);

f = scatteredInterpolant(x3,y3,z3);

Z3 = f(X3,Y3);

mesh(X3,Y3,Z3) %interpolated

plot3(x3,y3,z3,'.r','MarkerSize',15) %nonuniform
xlabel('Yaw 1 [degrees]')
ylabel('Yaw 2 [degrees]')
zlabel('Power [W]')