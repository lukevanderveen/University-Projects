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

// variable to keep track of current row being displayed
#define TIMER_PRESCALER 0 // prescaler for timer
#define TIMER_COMPARE_VALUE 160000 // compare value for timer

// debug function to print out the microBitFrameBuffer to verify correct LED placement
void printFrameBuffer() { 
    serial.printf("\r\nDisplay FrameBuffer:\r\n");
    
    // Loop through the 5x5 frame buffer
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            // Print each pixel value
            serial.printf("%d ", microBitDisplayFrameBuffer[row][col]);
        }
        serial.printf("\r\n");  // New line after each row
    }
}

// pin configuration function
void configPin(int pin, int port){
  if (port == 0){ // configure pins on port 0
    NRF_P0->PIN_CNF[pin] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                           (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                           (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                           (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                           (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
                           serial.printf("\r\npin configured = %d\r\n", pin);
  }else if(port == 1){ // configure pins on port 1
    NRF_P1->PIN_CNF[pin] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                           (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                           (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                           (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                           (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
                           serial.printf("\r\npin configured = %d\r\n", pin);
  }else {
    serial.printf("invalid pin\n");
  }
}

// clear microbit LEDs
void clearMicroBitDisplay() {
  memset(microBitDisplayFrameBuffer, 0, sizeof(microBitDisplayFrameBuffer));

  // set OUT pins for all rows and cols to high to clear the matrix
  NRF_P0->OUT |= (1 << 21) | (1 << 22) | (1 << 15) | (1 << 24) | (1 << 19) | (1 << 28) | (1 << 11) | (1 << 31) | (1 << 30);
  NRF_P1->OUT |= (1 << 5);
}

// update LEDs to give illusion of a static image
static void   microBitDisplayIsr() {
  const uint8_t rowPins[5] = {21, 22, 15, 24, 19}; // Pins for the 5 rows
  const uint8_t colPins[5] = {28, 11, 31, 5, 30}; // Pins for the 5 cols

  static uint8_t rowIndex = 0; 

  //deactivate previous row
  NRF_P0->OUT &= ~(1 << rowPins[rowIndex]);

  // update new row and wrap back to 0 once 4 is reached
  rowIndex = (rowIndex + 1) % 5;

  NRF_P0->OUT |= (1 << rowPins[rowIndex]); // activate new row

  //update cols based on frame buffer
  for (uint8_t col = 0; col < 5; col ++){
    if (microBitDisplayFrameBuffer[rowIndex][col] == 1){
      // activate colomn pixels
      if (col == 3){
        NRF_P1->OUT &= ~(1 << colPins[3]);
      } else {
        NRF_P0->OUT &= ~(1 << colPins[col]);
      }
    }else {
      // deactivate colomn pxiels
      if (col == 3){
        NRF_P1->OUT |= (1 << colPins[3]);
      } else {
        NRF_P0->OUT |= (1 << colPins[col]);
      }
    }
  }
  NRF_TIMER1->EVENTS_COMPARE[0] = 0; // reset event
}

// initalise microbit display
void initMicroBitDisplay() {
  //serial.printf("\r\ninit Initialising\r\n");
  clearMicroBitDisplay();

  // configure LED pins to output
  //  row 1              row 2              row 3            row 4            row 5
  configPin(21, 0); configPin(22, 0); configPin(15, 0); configPin(24, 0); configPin(19, 0);    
  //  col 1              col 2              col 3            col 4            col 5
  configPin(28, 0); configPin(11, 0); configPin(31, 0); configPin(5, 1); configPin(30, 0);          
  
  // drive rows and cols (except 4)
  NRF_P0->DIR |= (1 << 21) | (1 << 22) | (1 << 15) | (1 << 24) | (1 << 19) | (1 << 28) | (1 << 11) | (1 << 31) | (1 << 30);
  NRF_P1->DIR |= (1 << 5); // drive col 4 
  //serial.printf("\r\ninit Configured\r\n");

  NRFX_DELAY_MS(10); //short delay to ensure proper gpio set

  // timer setup for 100Hz
  //serial.printf("\r\ninit timer setting\r\n");
  NRF_TIMER_Type *timer = NRF_TIMER1;
  timer->MODE = TIMER_MODE_MODE_Timer;
  timer->PRESCALER = TIMER_PRESCALER;
  timer->CC[0] = TIMER_COMPARE_VALUE;
  timer->INTENSET = TIMER_INTENSET_COMPARE0_Set << TIMER_INTENSET_COMPARE0_Pos;

  NVIC_SetVector(TIMER1_IRQn, (uint32_t)microBitDisplayIsr); // set ISR for timer
  NVIC_EnableIRQ(TIMER1_IRQn); // enable timer

  timer->TASKS_CLEAR = 1; // clear timer
  timer->TASKS_START = 1; // start timer
}

// timer interrupt handler
void TIMER1_IRQHandler(void) {
    if (NRF_TIMER0->EVENTS_COMPARE[0]) {
        NRF_TIMER0->EVENTS_COMPARE[0] = 0; // Clear the compare event
        microBitDisplayIsr();              // update the display
    }
}

// set a pixel on the microbit
void setMicroBitPixel(uint8_t x, uint8_t y) {
  if (x < 5 && y < 5){ // validate for any x/y vals exceeding matrix size
    //serial.printf("\r\npixel set: x =%d y=%d\r\n", x, y);
    microBitDisplayFrameBuffer[y][x] = 1;
    //printFrameBuffer(); // debug 
    
  }
}

// clear a pixel on the microbit
void clearMicroBitPixel(uint8_t x, uint8_t y) {
  if (x < 5 && y < 5){ // validate whether x/y vals exceed matrix size
    //serial.printf("\r\npixel cleared: x =%d y=%d\r\n", x, y); 
    microBitDisplayFrameBuffer[y][x] = 0;
    //printFrameBuffer(); // debug
  } 
}


//###################################################################################
// Write your additional code for Subtask 2 below here (and leave these separators in your code!)
// put functions, global variables, consts, #DEFINEs etc. that you need for Subtask 2 here
//

// the bitmap in memory for the OLED pixels
// you must specify the type and the size, but keep the name the same
uint8_t oledDisplayFrameBuffer[8][128] = {0};

// clear all pixels on Oled display
void clearOledDisplay() {
  memset(oledDisplayFrameBuffer, 0, sizeof(oledDisplayFrameBuffer)); // clear frame buffer
  for (int page = 0; page < 8; page++){
    uint8_t commands[] = {
      static_cast<uint8_t>(0xB0 + page), // set page address
      0x00, // set lower colomn address
      0x10 // set higher colomn address
    }; // Set page and column addresses
    i2cExt.write(0x3C << 1, commands, sizeof(commands));
    uint8_t data[129];
    data[0] = 0x40; // data command
    memset(&data[1], 0, 128); // clear data
    i2cExt.write(0x3C << 1, data, 129); // write data to OLED
  }
}

// initialise OLED display
void initOledDisplay() {
  i2cExt.setFrequency(400000); //set I2C frequency
  
  // Send initialization sequence
  uint8_t initCommands[] = {
      0x80, 0xAE,       // Display OFF
      0x80, 0xD5, 0x80, // Clock divide ratio
      0x80, 0xA8, 0x3F, // Multiplex ratio
      0x80, 0xD3, 0x00, // Display offset
      0x80, 0x40,       // Start line
      0x80, 0x8D, 0x14, // Charge pump
      0x80, 0x20, 0x00, // Addressing mode
      0x80, 0xA1,       // Segment re-map
      0x80, 0xC8,       // COM scan direction
      0x80, 0xDA, 0x12, // COM pins
      0x80, 0x81, 0x7F, // Contrast
      0x80, 0xA4,       // Resume display from RAM
      0x80, 0xA6,       // Normal display
      0x80, 0xAF        // Display ON
  };

    i2cExt.write(0x3C << 1, initCommands, sizeof(initCommands)); // write commands to OLED
    clearOledDisplay(); // clear display and wait 0.05 seconds after to ensure this is done
    NRFX_DELAY_MS(50); // delay for initalisation
}



// set a single oled display pixel
void setOledPixel(uint8_t x, uint8_t y) {
  if (x < 128 && y < 64){ // boundary validation
    uint8_t page = y / 8; // vertical row of 8 pixels 
    uint8_t bit = y % 8;  // bit within the page  
    oledDisplayFrameBuffer[page][x] |= (1 << bit); // set pixel

    uint8_t commands[] = {
              0x80, static_cast<uint8_t>(0xB0 + page),    // Set page address
              0x80, static_cast<uint8_t>(x & 0x0F),       // Set lower column start address
              0x80, static_cast<uint8_t>(0x10 | (x >> 4)) // Set higher column start address
    };
    i2cExt.write(0x3C << 1, commands, sizeof(commands)); // write commands to I2C
    uint8_t data[] = {0x40, oledDisplayFrameBuffer[page][x]}; // Prepare data to write
    i2cExt.write(0x3C << 1, data, sizeof(data)); // write data to I2C
  }
}

// clear a single pixel 
void clearOledPixel(uint8_t x, uint8_t y) {
  if (x < 128 && y < 64){ // boundary validation 
    oledDisplayFrameBuffer[y / 8][x] &= ~(1 << (y % 8)); // clear pixel in frame buffer
    uint8_t commands[] = {
            static_cast<uint8_t>(0xB0 + (y / 8)), // Set page address
            0x00,                                // Set lower column start address
            0x10                                 // Set higher column start address
    };
    i2cExt.write(0x3C << 1, commands, sizeof(commands)); // write commands to ITC address
    i2cExt.write(0x3C << 1, oledDisplayFrameBuffer[x], 1); // write data to ITC address
  }
}

// draw a line on the oled display
void drawOledLine(uint8_t x_start, uint8_t y_start, uint8_t x_end, uint8_t y_end) {
  int dx = abs(x_end - x_start), sx = x_start < x_end ? 1 : -1; // calculate x increment
  int dy = -abs(y_end - y_start), sy = y_start < y_end ? 1 : -1; // calculate y increment
  int err = dx + dy, e2; // error value e_xy 

  while (true) {
      setOledPixel(x_start, y_start); // Set the pixel
      if (x_start == x_end && y_start == y_end) break; // exit if end reached
      e2 = 2 * err;
      if (e2 >= dy) { err += dy; x_start += sx; } // update x
      if (e2 <= dx) { err += dx; y_start += sy; } // update y
  }
}


//###################################################################################
// Write your additional code for Subtask 3 below here (and leave these separators in your code!)
// put functions, global variables, consts, #DEFINEs etc. that you need for Subtask 2 here
//
#define MODE_ACCELERATION 0 // default
#define MODE_JERK 1         // alternative
#define ACCELEROMETER_I2C_ADDRESS (0x19 << 1) // I2C for accelerometer

// mode A definitions
#define OLED_WIDTH 128           // Width of the OLED display
#define OLED_HEIGHT 64          // Height of the OLED display
#define ACCEL_MIN -512          // Min value of accelerometer for X-axis
#define ACCEL_MAX 511           // Max value of accelerometer for X-axis


uint8_t currentMode = MODE_ACCELERATION; // current mode of operation
int16_t prevAccelReading = 0; // Used to calculate jerk
uint8_t currentX = 0;         // Tracks the current column

//read x axis value
int8_t readAccelerometerX() {
  uint8_t highByte, lowByte;

  i2cInt.writeRegister(0x19 << 1, 0x20, 0x57); // Enable accelerometer (0x57)
  i2cInt.readRegister(0x19 << 1, 0x29, &highByte, 1); // Read high byte
  i2cInt.readRegister(0x19 << 1, 0x28, &lowByte, 1);  // Read low byte
  int16_t accelVal = ((int16_t) (highByte << 8) | lowByte); // combine bytees
  return (accelVal >> 6);
}

// Function to normalize accelerometer data for the OLED display (MODE A)
uint8_t normalizeAccelData(int16_t accelValue) {
  uint16_t min = -512; // min accelerometer val
  uint16_t max = 511; // max accelerometer value
  uint16_t range = max - min; // range of accelerometer values
  uint16_t offset = 512; // normalisation offset

  serial.printf("uncaledscaled value: %d\n", accelValue); // debug

  // clamp accelerometer value within range
  //if (accelValue < min) accelValue = min;
  //if (accelValue > max) accelValue = max;

  int16_t scaledVal = ((accelValue + offset) * 63) / range; // scale to 0-63

  serial.printf("scaled value: %d\n", scaledVal); // debug

  // if value exceeds boundaries, adjust to fit
  if (scaledVal < 0) scaledVal = 0;
  if (scaledVal > 63) scaledVal = 63;
  
  return static_cast<int8_t>(scaledVal);
}

void scrollGraph() {
  // Shift all columns left by one
  for (int x = 1; x < OLED_WIDTH; x++) {
    for (int page = 0; page < 8; page++) {
      oledDisplayFrameBuffer[page][x - 1] = oledDisplayFrameBuffer[page][x];
    }
  }

  // Clear the last column (rightmost)
  for (int page = 0; page < 8; page++) {
    oledDisplayFrameBuffer[page][OLED_WIDTH - 1] = 0x00;
  }
}

// Acceleration Mode
void modeA() {
  // set the microbit pixels in MODE A position
  setMicroBitPixel(0, 0);
  setMicroBitPixel(0, 1);
  setMicroBitPixel(0, 2);
  setMicroBitPixel(0, 3);
  setMicroBitPixel(0, 4);
  setMicroBitPixel(2, 2);

  uint8_t y_base = 63; // OLED display height is 64 pixels, so bottom is 63
  static uint8_t x_pos = 0;   // start at 0, then increment across

  uint16_t currentAccel = readAccelerometerX();
  uint8_t normalised = normalizeAccelData(currentAccel); // normalise X axis 

  serial.printf("normalised value: %d\n", normalised);

  uint8_t y_top = y_base - normalised; // calculate line position

  drawOledLine(x_pos, y_base, x_pos, y_top); // draw line

  x_pos++; // move to next colomn
 
  scrollGraph(); // scroll the display

  int page = normalised / 8; // page to update
  int bitPosition = normalised % 8; // bit position

  oledDisplayFrameBuffer[page][OLED_WIDTH - 1] = (1 << bitPosition); // set pixel in last column

  // focus on last page
  // update OLED display
  uint8_t commands[] = {
    0x80, static_cast<uint8_t>(0xB0 + page),  // Set page address
    0x80, 0x00,                               // Set lower column start address
    0x80, 0x10                                // Set higher column start address
  };
  i2cExt.write(0x3C << 1, commands, sizeof(commands)); // Write commands to I2C
  i2cExt.write(0x3C << 1, &oledDisplayFrameBuffer[page][0], 128); // Write pixel data  

  NRFX_DELAY_MS(50); // delay for update
}

//
void graphData(uint8_t refreshRate){
  // initalise microbit pixels
  initOledDisplay();

  // MICROBIT LEDs in default mode
  initMicroBitDisplay();
  while(true){
    modeA();

    // Delay for the specified refresh rate
    NRFX_DELAY_MS(1000 / refreshRate);  // Adjust the delay for refresh rate
  }
}

/*
LEGACY


*/