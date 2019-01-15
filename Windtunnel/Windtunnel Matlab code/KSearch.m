close all
clear all
clc

%% Connecting with Arduino
instrreset
arduinoSerial = serial('COM7','BAUD', 9600);
connectToArduino(arduinoSerial)

%% Testing connection with Arduino
disp('Starting test measurement.');
[RPM1, measuredCurrent1] = setCurrent1GetRPM1AndCurrent1(arduinoSerial,30);
disp(['Test RPM1: ',num2str(RPM1),' RPM. Test Current 1: ',num2str(measuredCurrent1),' mA.']);

disp('Please check the connections. Press any key to continue.');
pause

%% Setting up variables
Istart = 0;
Iend = 500;
Istep = 5;                             %[mA]
iterations = ((Iend-Istart)/Istep)+1;
KT=3.52E-3; %Nm A^-1
data = zeros(iterations,4);

%% Setting windtunnel initial conditions
%Setting initial current
sendToArduino(arduinoSerial, 5, Istart)
disp(['Set initial Current1 to ',num2str(Istart),' mA']);

%% Starting measurements
tic;
for i = 1:iterations
    
    %display progress
    disp(['Starting test experiment run ',num2str(i),'/',num2str(iterations),'. Time passed is ',num2str(toc),' seconds.']);
    
    %measure RPM1 and Current1 and compute power
    [RPM1, measuredCurrent1] = setCurrent1GetRPM1AndCurrent1(arduinoSerial,((i-1)*Istep + Istart));
    Power1 = measuredCurrent1 * KT * RPM1 *pi/30;
    
    %Save data
    data(i,1) = (i-1)*Istep+Istart;     %imposed current
    data(i,2) = RPM1;                   %measured RPM
    data(i,3) = measuredCurrent1;       %measured current
    data(i,4) = Power1;                 %computed power
    
    %display progress
    disp(['Imposed Current1 is: ',num2str((i-1)*Istep+Istart),' mA. Actual current is: ',num2str(measuredCurrent1), ' mA. Actual RPM1 is: ',num2str(RPM1),' RPM.']);
    disp(['Measured Power1 is: ',num2str(Power1),' W. Time passed is: ',num2str(toc), ' seconds.']) ;
    
    %Stop if the turbine starts turning the wrong way
    if Power1 < 0
        disp('Negative power, stopped the loop.');
        data( all(~data,2), : ) = [];
        break
    end
end

%set current to 0 to allow free rotation
sendToArduino(arduinoSerial, 5, 0)
disp('Finished! Set current to 0 to allow free rotation.');

%% Plotting data
figure
f = fit(data(:,3), data(:,4),'smoothingspline');
plot(f,'b')
hold on

% plot(data(:,3), data(:,4), 'o')
err = .0035*ones(size(data(:,4)));
errorbar(data(:,3),data(:,4),err,'s','MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor','red');
plot(0.3171,0.1902, '*','MarkerSize',15);



legend({'curve fit','Measurement points with \sigma=.0035 [W]'},'fontSize',14)
xlabel('Measured current [mA]','fontSize',14)
ylabel('Generated power [W]','fontSize',14)
axis([0 max(data(:,3))*1.1 0 max(data(:,4))*1.1])



%% Computing K
[maxPower, maxPowerPosition] = max(data(:,4));
ImaxPower = data(maxPowerPosition,3);
ImaxPower = 0.2028;

% RPMmaxPower = data(maxPowerPosition,2);
         %convert RPM to rad/s
K = ImaxPower/(radmaxPower^2);
 
RPMmaxPower = 2042.87;
radmaxPower = RPMmaxPower*pi/30;
Ides = Kdes * radmaxPower^2


disp(['K is computed to be ',num2str(K),' A * s^2 * rad^-2']);