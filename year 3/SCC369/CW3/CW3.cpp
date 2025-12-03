#include "MicroBit.h"

//
// references to external objects
//
extern NRF52Serial serial;  // serial object for displaying type of sensor and debugging 
extern NRF52I2C    i2cInt;  // internal I2C object for talking to accelerometer
extern NRF52I2C    i2cExt;  // external I2C object for talking to OLED display

//###################################################################################
// Write your code for Subtask 1 below here (and leave these separators in your code!)
// put functions, global variables, consts, #DEFINEs etc. that you need for Subtask 1 here
//

// the bitmap in memory for the LED pixels
// you can change the type and the size if necessary, but keep the name the same
uint8_t microBitDisplayFrameBuffer[5][5] = {0};

// 
static void microBitDisplayIsr() {

  // ISR code 

}

//
void initMicroBitDisplay() {

}

//
void clearMicroBitDisplay() {

}

//
void setMicroBitPixel(uint8_t x, uint8_t y) {

}

//
void clearMicroBitPixel(uint8_t x, uint8_t y) {

}


//###################################################################################
// Write your additional code for Subtask 2 below here (and leave these separators in your code!)
// put functions, global variables, consts, #DEFINEs etc. that you need for Subtask 2 here
//

// the bitmap in memory for the OLED pixels
// you must specify the type and the size, but keep the name the same
uint8_t oledDisplayFrameBuffer[0][0];

//
void initOledDisplay() {

}

//
void clearOledDisplay() {

}

//
void setOledPixel(uint8_t x, uint8_t y) {

}

//
void clearOledPixel(uint8_t x, uint8_t y) {

}

//
void drawOledLine(uint8_t x_start, uint8_t y_start, uint8_t x_end, uint8_t y_end) {

}


//###################################################################################
// Write your additional code for Subtask 3 below here (and leave these separators in your code!)
// put functions, global variables, consts, #DEFINEs etc. that you need for Subtask 2 here
//

//
void graphData(uint8_t refreshRate){

}
