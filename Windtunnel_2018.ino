//Loading Libraries
#include <Stepper.h>
#include <MultiStepper.h>
#include <AccelStepper.h>
#include <Servo.h>

//Defining constants
const int maxWindAngle = 13; //max windAngle before blade hits tunnel
const int maxYawAngle = 80;
const int h = 2380; //hight [steps]
const int H = 235; //hight [mm]
const float C = h / H; // [steps]/[mm]= ratio between steps and mm
const int L = 500; //Horizontal distance between turbines [mm]
const float pi = 3.14159265359;
const float K = 2.8e-06; //The value of the optimal constant for induction based rotor speed control

// Calibrating the Servo's to 0 degrees relative to the wind.
const float angleServo1 = 90;
const float angleServo2 = 90;
const float angleServo3 = 90;

//Defining pins
int turbinePin1 = 11;
int turbinePin2 = 12;
int turbinePin3 = 13;
int turbine1On = 48;
int turbine2On = 50;
int turbine3On = 52;

int RESETPIN = 46;

//  Defining servo objects for the library
Servo servo1;
Servo servo2;
Servo servo3;

// Attaching steppers
AccelStepper stepper1(1, 7, 6); //pin 2 = step, pin 3 = direction
AccelStepper stepper2(1, 5, 4); //pin 4 = step, pin 5 = direction
AccelStepper stepper3(1, 3, 2); //pin 6 = step, pin 7 = direction
MultiStepper steppers; //Using MultiStepper function to make sure they move simultaneously
long StepperPositions[3];

//Defining input variables
int windAngleDeg = 0; //Setting the angle of the wind to 0 degrees

float yawAngle1 = 0; //Setting the start position of the yawAngle to 0 degrees relative to the wind.
float yawAngle2 = 0;
float yawAngle3 = 0;

float inputCurrent1 = 0; //Setting the initial torque to 0
float inputCurrent2 = 0;
float inputCurrent3 = 0;

float induction1 = 1; //Setting the initial value of the axial induction multiplier to 1
float induction2 = 1;
float induction3 = 1;

int runGreedyControl = 0; //Setting the initial state of the torque control algorithm to 'off'

int state = 0; //Setting the initial state to the 'Read' state

int value = 0; //Defining the read variables
int value1 = 0;
int value2 = 0;
int value3 = 0;
float value4 = 1;
float value5 = 1;
float value6 = 1;

unsigned long startMillis = 0; //Defining timer variable

//Defining output variables
float RPM1 = 0; //Angular velocity in RPM
float RPM2 = 0;
float RPM3 = 0;

float rad1 = 0; //Angular velocity in rad/s
float rad2 = 0;
float rad3 = 0;

float measuredCurrent1 = 0; //measuredCurrent in µA
float measuredCurrent2 = 0;
float measuredCurrent3 = 0;

void setup() {
  // Serial start
  Serial.begin(9600);

  //Defining stepper objects for the stepper library
  steppers.addStepper(stepper1);
  steppers.addStepper(stepper2);
  steppers.addStepper(stepper3);

  //Limiting the speed of the stepper motors
  stepper1.setMaxSpeed(100);
  stepper2.setMaxSpeed(100);
  stepper3.setMaxSpeed(100);

  //Defining pinModes
  pinMode(turbinePin1, OUTPUT);
  pinMode(turbinePin2, OUTPUT);
  pinMode(turbinePin3, OUTPUT);
  pinMode(RESETPIN, OUTPUT);

  //Attaching servos
  servo1.attach(8);
  servo2.attach(9);
  servo3.attach(10);

  //Let the servos start at -30 degrees relative to the wind to make sure they don't fall on the blades
  servo1.write(60);
  servo2.write(60);
  servo3.write(60);

  //ControlledFall
  digitalWrite(RESETPIN, HIGH);
  delay(4000);
  digitalWrite(RESETPIN, LOW);

  //Setting the wind angle to 0 degrees and the yaw angles to 0 degrees
  windAngle(); 
  voidYawAngle1();
  voidYawAngle2();
  voidYawAngle3();
  
  //setting initial turbine torque
  //map Matlab command from [0 - 300 mA] to analogWrite [0 - 255] {10% = 0 mA, 90% = 300 mA}
  inputCurrent1 = (map(constrain(inputCurrent1, 0, 300), 0, 300, 25, 230));
  inputCurrent2 = (map(constrain(inputCurrent2, 0, 300), 0, 300, 25, 230));
  inputCurrent3 = (map(constrain(inputCurrent3, 0, 300), 0, 300, 25, 230));
  analogWrite(turbinePin1, inputCurrent1);
  analogWrite(turbinePin2, inputCurrent2);
  analogWrite(turbinePin3, inputCurrent3);

  //turning on turbines
  digitalWrite(turbine1On, HIGH);
  digitalWrite(turbine2On, HIGH);
  digitalWrite(turbine3On, HIGH);
  Serial.println(String("Done;"));
}

void loop() {

  // read input
  if (state == 0) {
    serialInput();
  }

  // Set yaw 1
  if (state == 1) {
    yawAngle1  = value;
    voidYawAngle1();

    state = 0;
    runGreedyControl = 0;
    Serial.println(String("Done;"));
  }

  // Set yaw 2
  else if (state == 2) {
    yawAngle2  = value;
    voidYawAngle2();

    state = 0;
    runGreedyControl = 0;
    Serial.println(String("Done;"));
  }

  // Set yaw 3
  else if (state == 3) {
    yawAngle3  = value;
    voidYawAngle3();

    state = 0;
    runGreedyControl = 0;
    Serial.println(String("Done;"));
  }

  // Set wind angle
  else if (state == 4) {
    windAngleDeg = value;
    windAngle();

    state = 0;
    runGreedyControl = 0;
    Serial.println(String("Done;"));
  }

  //Set current1 and send RPM1 for determining K-value
  else if (state == 5) {

    // Set value from serialread to inputcurrent1
    inputCurrent1 = value;
    inputCurrent2 = 0;
    inputCurrent3 = 0;

    //map Matlab command from [0 - 300 mA] to analogWrite [0 - 255] {10% = 0 mA, 90% = 300 mA}
    inputCurrent1 = (map(constrain(inputCurrent1, 0, 300), 0, 300, 25, 230));
    inputCurrent2 = (map(constrain(inputCurrent2, 0, 300), 0, 300, 25, 230));
    inputCurrent3 = (map(constrain(inputCurrent3, 0, 300), 0, 300, 25, 230));

    // setting current
    analogWrite(turbinePin1, inputCurrent1);
    analogWrite(turbinePin2, inputCurrent2);
    analogWrite(turbinePin3, inputCurrent3);

    //Reset RPM and measuredCurrent for average counter
    RPM1 = 0;
    measuredCurrent1 = 0;
    
    delay(7000); //Wait for wind to settle

    //measure currents and RPM's
    int loops = 1000;
    for (int i = 0; i <= loops; i++) { 
      //Read RPM and current and map analog input from [0 - 4V] to RPM and current [µA]
      RPM1 += map(analogRead(6), 0, 818, -5000, 5000); //Sum over 1000 measurements
      measuredCurrent1 += map(analogRead(5), 0, 818, 0, 300000); // 300000 µA is chosen because it gives a higher resolution than 300 mA, and is intuitive to calculate with
    }

    // Take average over 1000 measurements
    RPM1 = RPM1 / loops;
    measuredCurrent1 = measuredCurrent1 / loops;

    //Print RPM1 and current1
    Serial.println(String(String("RPM: ") + String(RPM1) + String(";")));
    Serial.println(String(String("Current: ") + String(measuredCurrent1) + String(";")));
    
    state = 0;
    runGreedyControl = 0;
  }

  //set yaw1, yaw2 and yaw3, run greedy torque control for 8 seconds, read and take average over 1000 measurements, and send rpm1, rpm2 and rpm3 and current1, current2 and current3
  else if (state == 6) {

    //set yaw1, yaw2 and yaw3
    yawAngle1 = value1;
    yawAngle2 = value2;
    yawAngle3 = value3;
    voidYawAngle1();
    voidYawAngle2();
    voidYawAngle3();

    //start timer
    startMillis = millis();

    //Reset RPM and measuredCurrent for average counter
    RPM1 = 0;
    RPM2 = 0;
    RPM3 = 0;
    measuredCurrent1 = 0;
    measuredCurrent2 = 0;
    measuredCurrent3 = 0;
    
    //Run greedy torque controller and wait for wind to settle
    while (millis() - startMillis < 8000) {
      greedyControl();
    }

    //Measure currents and RPM's
    int loops = 1000;
    for (int i = 0; i <= loops; i++) {  
      //Read RPM's and currents and map analog input from [0 - 4V] to RPM and current [µA]
      RPM1 += map(analogRead(6), 0, 818, -5000, 5000); //Sum over 1000 measurements
      RPM2 += map(analogRead(4), 0, 818, -5000, 5000);
      RPM3 += map(analogRead(2), 0, 818, -5000, 5000);
      measuredCurrent1 += map(analogRead(5), 0, 818, 0, 300000);
      measuredCurrent2 += map(analogRead(3), 0, 818, 0, 300000);
      measuredCurrent3 += map(analogRead(1), 0, 818, 0, 300000);
      greedyControl(); //run 1 step of the greedy torque controller
    }

    // Take average over 1000 measurements
    RPM1 = RPM1 / loops;
    RPM2 = RPM2 / loops;
    RPM3 = RPM3 / loops;
    measuredCurrent1 = measuredCurrent1 / loops;
    measuredCurrent2 = measuredCurrent2 / loops;
    measuredCurrent3 = measuredCurrent3 / loops;

    //Print RPM's and currents
    Serial.println(String(String("RPM:") + String(RPM1) + String(";") + String(RPM2) + String(";") + String(RPM3) + String(";")));
    Serial.println(String(String("Current:") + String(measuredCurrent1) + String(";") + String(measuredCurrent2) + String(";") + String(measuredCurrent3) + String(";")));

    state = 0;
    runGreedyControl = 1;
    induction1 = 1; //Make sure the induction factor is 1
    induction2 = 1;
    induction3 = 1;
  }

  //set yaw1, yaw2 and yaw3 and read and take average over 100 measurements, and send rpm1, rpm2 and rpm3 and current1, current2 and current3
  else if (state == 7) {
    
    //set yaw1, yaw2 and yaw3
    yawAngle1 = value1;
    yawAngle2 = value2;
    yawAngle3 = value3;
    voidYawAngle1();
    voidYawAngle2();
    voidYawAngle3();

    //Reset RPM and measuredCurrent for average counter
    RPM1 = 0;
    RPM2 = 0;
    RPM3 = 0;
    measuredCurrent1 = 0;
    measuredCurrent2 = 0;
    measuredCurrent3 = 0;

    //measure currents and RPM's
    int loops = 100;
    for (int i = 0; i <= loops; i++) {
      //read RPM's and currents and map analog input from [0 - 4V] to RPM and current [µA]
      RPM1 += map(analogRead(6), 0, 818, -5000, 5000); //Sum over 100 measurements
      RPM2 += map(analogRead(4), 0, 818, -5000, 5000);
      RPM3 += map(analogRead(2), 0, 818, -5000, 5000);

      measuredCurrent1 += map(analogRead(5), 0, 818, 0, 300000);
      measuredCurrent2 += map(analogRead(3), 0, 818, 0, 300000);
      measuredCurrent3 += map(analogRead(1), 0, 818, 0, 300000);
      greedyControl(); //run 1 step of the greedy torque controller
    }

    // Take average over 100 measurements
    RPM1 = RPM1 / loops;
    RPM2 = RPM2 / loops;
    RPM3 = RPM3 / loops;
    measuredCurrent1 = measuredCurrent1 / loops;
    measuredCurrent2 = measuredCurrent2 / loops;
    measuredCurrent3 = measuredCurrent3 / loops;

    //print currents and rpms
    Serial.println(String(String("RPM:") + String(RPM1) + String(";") + String(RPM2) + String(";") + String(RPM3) + String(";")));
    Serial.println(String(String("Current:") + String(measuredCurrent1) + String(";") + String(measuredCurrent2) + String(";") + String(measuredCurrent3) + String(";")));

    state = 0;
    runGreedyControl = 1;
    induction1 = 1; //Make sure the induction factor is 1
    induction2 = 1;
    induction3 = 1;
  }

  // Turn off the Enable pins of the Escon drivers
  else if (state == 8) {
    if (value < 1) {
      digitalWrite (turbine1On, LOW);
      digitalWrite (turbine2On, LOW);
      digitalWrite (turbine3On, LOW);
    }
    else {
      digitalWrite (turbine1On, HIGH);
      digitalWrite (turbine2On, HIGH);
      digitalWrite (turbine3On, HIGH);
    }
    state = 0;
    runGreedyControl = 0;
  }

  //set yaw1, yaw2, yaw3, induction1, induction2, induction3, run greedy current for 8 seconds and send rpm1, rpm2, rpm3 and current1, current2 and current3
  else if (state == 9) {

    //set yaw1, yaw2 and yaw3
    yawAngle1 = value1;
    yawAngle2 = value2;
    yawAngle3 = value3;
    voidYawAngle1();
    voidYawAngle2();
    voidYawAngle3();

    //set axial induction multipliers
    induction1 = value4;
    induction2 = value5;
    induction3 = value6;

    //start timer
    startMillis = millis();

    //Reset RPM and measuredCurrent for average counter
    RPM1 = 0;
    RPM2 = 0;
    RPM3 = 0;
    measuredCurrent1 = 0;
    measuredCurrent2 = 0;
    measuredCurrent3 = 0;

    //Run greedy torque controller and wait for wind to settle
    while (millis() - startMillis < 7000) {
      greedyControl();
    }

    int loops = 1000;
    for (int i = 0; i <= loops; i++) {
      //read RPM's and currents and map analog input from [0 - 4V] to RPM and current [µA]
      RPM1 += map(analogRead(6), 0, 818, -5000, 5000); //Sum over 1000 measurements
      RPM2 += map(analogRead(4), 0, 818, -5000, 5000);
      RPM3 += map(analogRead(2), 0, 818, -5000, 5000);

      measuredCurrent1 += map(analogRead(5), 0, 818, 0, 300000);
      measuredCurrent2 += map(analogRead(3), 0, 818, 0, 300000);
      measuredCurrent3 += map(analogRead(1), 0, 818, 0, 300000);
      greedyControl(); //run 1 step of the greedy torque controller
    }

    // Take average over 1000 measurements
    RPM1 = RPM1 / loops;
    RPM2 = RPM2 / loops;
    RPM3 = RPM3 / loops;
    measuredCurrent1 = measuredCurrent1 / loops;
    measuredCurrent2 = measuredCurrent2 / loops;
    measuredCurrent3 = measuredCurrent3 / loops;

    //print RPM's and currents
    Serial.println(String(String("RPM:") + String(RPM1) + String(";") + String(RPM2) + String(";") + String(RPM3) + String(";")));
    Serial.println(String(String("Current:") + String(measuredCurrent1) + String(";") + String(measuredCurrent2) + String(";") + String(measuredCurrent3) + String(";")));

    state = 0;
    runGreedyControl = 1;
  }

//Return to read state for weird inputs
  else {
    state = 0;
  }
}

void serialInput() {
  
  //wait for user input
  while (!Serial.available()) {
    if (runGreedyControl > 0) {
      greedyControl(); //run steps of the greedy torque controller while waiting for user input
    }
  }

  while (Serial.available() > 0) {
    Serial.flush(); //Wait for the transmission of outgoing serial data to complete.
    state = Serial.readStringUntil(';').toFloat();    //splits serial input string
    
    if ((state == 1) || (state == 2) || (state == 3) || (state == 4) || (state == 5) || (state == 8)) {
      value = Serial.readStringUntil(';').toFloat(); //Read only 1 input
    }
    else if ((state == 6) || (state == 7)) {
      value1 = Serial.readStringUntil(';').toFloat(); //Read 3 inputs
      value2 = Serial.readStringUntil(';').toFloat();
      value3 = Serial.readStringUntil(';').toFloat();
    }
    else  if (state == 9) {
      value1 = Serial.readStringUntil(';').toFloat(); //Read 6 inputs
      value2 = Serial.readStringUntil(';').toFloat();
      value3 = Serial.readStringUntil(';').toFloat();
      value4 = Serial.readStringUntil(';').toFloat();
      value5 = Serial.readStringUntil(';').toFloat();
      value6 = Serial.readStringUntil(';').toFloat();
    }
    else {
      state = 0; //return to read state for weird inputs
    }
  }
}

void greedyControl() {
  // read RPM's and convert to rad/s
  rad1 = map(analogRead(6), 0, 818, -5000, 5000) / 30 * pi;
  rad2 = map(analogRead(4), 0, 818, -5000, 5000) / 30 * pi;
  rad3 = map(analogRead(2), 0, 818, -5000, 5000) / 30 * pi;

  // compute new inputCurrents
  inputCurrent1 = map(constrain((K * induction1 * rad1 * rad1 * 1000), 0, 300), 0, 300, 25, 230); //[*1000 because of mA]
  inputCurrent2 = map(constrain((K * induction2 * rad2 * rad2 * 1000), 0, 300), 0, 300, 25, 230);
  inputCurrent3 = map(constrain((K * induction3 * rad3 * rad3 * 1000), 0, 300), 0, 300, 25, 230);

  //write currents
  analogWrite(turbinePin1, inputCurrent1);
  analogWrite(turbinePin2, inputCurrent2);
  analogWrite(turbinePin3, inputCurrent3);
}

void windAngle() {
  //Bound wind angle
  if (windAngleDeg > maxWindAngle) { 
    windAngleDeg = maxWindAngle;
  }
  else if (windAngleDeg < -maxWindAngle) {
    windAngleDeg = -maxWindAngle;
  }

  //compute individual positions
  StepperPositions[0] = h / 2 + L * tan(windAngleDeg * pi / 180) * C;
  StepperPositions[1] = h / 2;
  StepperPositions[2] = h / 2 - L * tan(windAngleDeg * pi / 180) * C;

   //move steppers
  steppers.moveTo(StepperPositions);
  steppers.runSpeedToPosition();
}

void voidYawAngle1() {
    //Bound yaw angle
  if (yawAngle1 > maxYawAngle) {
    yawAngle1 = maxYawAngle;
  }
  else if (yawAngle1 < -maxYawAngle) {
    yawAngle1 = -maxYawAngle;
  }
  servo1.write(angleServo1 + yawAngle1);
}

void voidYawAngle2() {
  //Bound yaw angle
  if (yawAngle2 > maxYawAngle) {
    yawAngle2 = maxYawAngle;
  }
  else if (yawAngle2 < -maxYawAngle) {
    yawAngle2 = -maxYawAngle;
  }
  servo2.write(angleServo2 + yawAngle2);
}

void voidYawAngle3() {
  //Bound yaw angle
  if (yawAngle3 > maxYawAngle) {
    yawAngle3 = maxYawAngle;
  }
  else if (yawAngle3 < -maxYawAngle) {
    yawAngle3 = -maxYawAngle;
  }
  servo3.write(angleServo3 + yawAngle3);
}
