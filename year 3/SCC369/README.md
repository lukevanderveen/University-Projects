# 369: Embedded Systems

This module was personally one of my favorites and also one of the most challenging, it involved working on extremely 
low level coding, working with registers in C++. We would work with the Microbit and acompanying hardware, as well as
its specification to configure GPIO pins.
The files that I would work on are the ones formatted in the following way CWX.cpp or MainSubtaskRunnerCWX.cpp, X standing 
for the current coursework eg CW1, CW2, CW3

## CW1
- A counter in bits
- Every time you pressed the 2 buttons it would add or subtract 1 depending on the button to an 5 bit number.
- This number was displayed on the first row of the 5x5 light matrix on the microbit and was displayed in binary
- As well as this it also displayed voltage through the this row based on a circuit and a rotational switch 

## CW2
- Create an internal clock for the microbit using electrical frequency
- Create a print function using serial bit banging and the internal clock, converts bits and electrical siganls to chars
- Create and accelerometer, that uses the new print fucntion to produce XYZ coordinates on where the microbit is in space
- Make a noise using the GPIO pins and the accelerometer's coordinates, higher frequency based on higher Y cord

## CW3
- Utilise a OLED display circuit to produce a single dot in the middle of the display
- Create continuous line through the centre of the screen
- Read the accelerometer and display values in the form of a continuously sliding bar graph set from the bottom
  of the screen as mode A
- Mode B would be the same but done from the centre so it would look like frequency display
