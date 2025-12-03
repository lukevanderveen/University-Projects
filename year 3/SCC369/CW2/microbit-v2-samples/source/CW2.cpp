#include "MicroBit.h"

/*
commands: move MICROBIT.hex D:\
*/
//###################################################################################
// Write your code for Subtask 1 here (and leave these separators in your code!):

//start a timer code
// this timer accounts for the slight overhead of processing 
void startTimerContinuous(uint32_t duration) {
    NRF_TIMER0->MODE = TIMER_MODE_MODE_Timer; // set the timer to Timer mode
    NRF_TIMER0->PRESCALER = 0;               // 16 MHz clock
    NRF_TIMER0->BITMODE = TIMER_BITMODE_BITMODE_16Bit << TIMER_BITMODE_BITMODE_Pos; // 16-bit mode
    NRF_TIMER0->CC[0] = duration;            // set the compare value 
    NRF_TIMER0->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Msk; // auto-clear on compare match
    NRF_TIMER0->TASKS_CLEAR = 1;             // clear the timer
    NRF_TIMER0->TASKS_START = 1;             // start the timer in continuous mode
}

void waitForNextBit() {
    while(NRF_TIMER0->EVENTS_COMPARE[0] == 0); // wait for the compare event
    NRF_TIMER0->EVENTS_COMPARE[0] = 0;         // clear the compare event
}

//
void bitBangSerial(char *string) {
    static bool initalised  = false;
    if (!initalised) {
        NRF_P0->PIN_CNF[6] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);

        NRF_P0->PIN_CNF[4] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) | 
                             (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                             (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                             (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                             (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);

        initalised = true;
    }
    const float bitTime = (1.0 / 115200) * 1e6; //calculate bit time in microseconds
    uint32_t duration = (uint32_t)(bitTime * 16); 

    startTimerContinuous(duration);

    while (*string) {
        char data = *string++;
        NRF_P0->OUTCLR = (1<<6) | (1<<4); //set pin 6 and 4 to low
        waitForNextBit();
        //send bits data LSB first
        for (int i = 0; i < 8; i++){
            if (data & (1<<i)){
                NRF_P0->OUTSET = (1<<6) | (1<<4);
                waitForNextBit();
            }else {
                NRF_P0->OUTCLR = (1<<6) | (1<<4);
                waitForNextBit();
            }
        }
        NRF_P0->OUTSET = (1<<6) | (1<<4); 
        waitForNextBit();
    }
}

//
void voteForChocolate(void) {
    const char *chocolate = "Freddo\n\r";
    //const char *chocolate = "A"; // simple single ascii character for testing
    startTimerContinuous(240000); 
    while (1) {
        bitBangSerial((char *)chocolate);
        waitForNextBit();   
    }  
        
}

//###################################################################################
// Write your additional code for Subtask 2 here:

// initalise pins and registers for accelerometer
    void initAccelerometer() {
    // Initialize the TWI peripheral
    NRF_P0->PIN_CNF[8] = (GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos);

    NRF_P0->PIN_CNF[16] = (GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos);

    NRF_TWI0->PSEL.SCL = P0_08; // SCL pin 
    NRF_TWI0->PSEL.SDA = P0_16; // SDA pin 
    NRF_TWI0->FREQUENCY = TWI_FREQUENCY_FREQUENCY_K100 << TWI_FREQUENCY_FREQUENCY_Pos;
    NRF_TWI0->ADDRESS = 0x19; // Accelerometer I2C address
    NRF_TWI0->ENABLE = TWI_ENABLE_ENABLE_Enabled << TWI_ENABLE_ENABLE_Pos;

    // Configure LSM303AGR
    uint8_t configData[2] = {0x20, 0x57}; // CTRL_REG1_A, Normal mode, all axes enabled
    
    NRF_TWI0->TASKS_STARTTX = 1; //initiate start task
    NRF_TWI0->TXD = configData[0]; //send  ctrl reg address
    while (!NRF_TWI0->EVENTS_TXDSENT); // loop till it has been recieved
    NRF_TWI0->EVENTS_TXDSENT = 0;

    NRF_TWI0->TXD = configData[1]; // send normal power mode 
    while (!NRF_TWI0->EVENTS_TXDSENT);
    NRF_TWI0->EVENTS_TXDSENT = 0;

    NRF_TWI0->TASKS_STOP = 1; // end task
    while (!NRF_TWI0->EVENTS_STOPPED); // wait till it flags that event has been stopped successfully
    NRF_TWI0->EVENTS_STOPPED = 0;
}

//read accelerometer axis (fed either a high or low register)
int16_t readAccelerometerAxis(uint8_t axisRegister) {
    NRF_TWI0->TASKS_STARTTX = 1;
    NRF_TWI0->TXD = axisRegister;
    while (!NRF_TWI0->EVENTS_TXDSENT);
    NRF_TWI0->EVENTS_TXDSENT = 0;

    NRF_TWI0->TASKS_STARTRX = 1;
    while (!NRF_TWI0->EVENTS_RXDREADY);
    NRF_TWI0->EVENTS_RXDREADY = 0;

    NRF_TWI0->TASKS_STOP = 1;
    while (!NRF_TWI0->EVENTS_STOPPED);
    NRF_TWI0->EVENTS_STOPPED = 0;

    return NRF_TWI0->RXD;
}


int getAccelerometerSample(char axis) {
    static bool AccelerometerInitialized = false;
    uint8_t axisRegisterLow;
    uint8_t axisRegisterHigh;


    switch (axis) {
        case 'X':
            axisRegisterLow = 0x28; // X out register low
            axisRegisterHigh = 0x29; // X out register high
            break;
        case 'Y':
            axisRegisterLow = 0x2A; // Y out register low
            axisRegisterHigh = 0x2B; // Y out register high
            break;
        case 'Z':
            axisRegisterLow = 0x2C; // Z out register low
            axisRegisterHigh = 0x2D; // Z out register high
            break;
        default:
            return 0; // axis invalid
    }

    if (!AccelerometerInitialized) { //initilise accelerometer
        initAccelerometer();
        AccelerometerInitialized = true;
    }

    uint8_t rawValueLow = readAccelerometerAxis(axisRegisterLow); // read value from low
    uint8_t rawValueHigh = readAccelerometerAxis(axisRegisterHigh); // read value from high

    int16_t combinedValue = ((int16_t) (rawValueHigh << 8) | rawValueLow); //combine both the high and the low value
 
    return (combinedValue >> 6); //scale for ranges
}

//
void showAccelerometerSamples(void) {
    const char *sub2 = "Subtask 2 starting\n";
    bitBangSerial((char *) sub2);
    char xBuffer[10], yBuffer[10], zBuffer[10]; //buffers to store x y z values
    startTimerContinuous(200000); // Delay 
    int x, y, z;
    while (1) {
        x = getAccelerometerSample('X');
        y = getAccelerometerSample('Y');
        z = getAccelerometerSample('Z');

        waitForNextBit();
        const char *xPrefix = "[X: ";
        bitBangSerial((char *) xPrefix);
        itoa(x, xBuffer);
        bitBangSerial(xBuffer);
        
        waitForNextBit();
        const char *yPrefix = "] [Y: ";
        bitBangSerial((char *) yPrefix);
        itoa(y, yBuffer);
        bitBangSerial(yBuffer);
        
        waitForNextBit();
        const char *zPrefix = "] [Z: ";
        bitBangSerial((char *) zPrefix);
        itoa(z, zBuffer);
        bitBangSerial(zBuffer);
        
        waitForNextBit();
        const char *endOfLine = "]\r\n";
        bitBangSerial((char *) endOfLine);  
        startTimerContinuous(200000); // longer wait after printing to clear the bit bang serial otherwise it produces nonsense output
        waitForNextBit();
    }
}

//###################################################################################
// Write your additional code for Subtask 3 here:

//
void makeNoise(void) {
    const char *sub3 = "... Subtask 3 running ... \n";
    bitBangSerial((char *) sub3);

    // define a one-entry ‘sequence’ which applies to all PWM0 channels
    static uint16_t pwm_seq = 8000; // e.g. 8k ticks
    NRF_PWM0->SEQ[0].PTR = (uint32_t) &pwm_seq; // use for sequence[0]
    NRF_PWM0->SEQ[0].CNT = 1; // just 1 entry

    //PWM out pin configuration
    NRF_P0->PIN_CNF[0] = (GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos);

    NRF_PWM0->PSEL.OUT[0] = P0_00; // set the out to the speaker pin

    NRF_PWM0->ENABLE = PWM_ENABLE_ENABLE_Enabled << PWM_ENABLE_ENABLE_Pos; // enable PWM

    // configure behaviour of PWM
    NRF_PWM0->MODE = PWM_MODE_UPDOWN_Up << PWM_MODE_UPDOWN_Pos; 
    NRF_PWM0->PRESCALER = PWM_PRESCALER_PRESCALER_DIV_16 << PWM_PRESCALER_PRESCALER_Pos; // divide by 16 (1MHz)
    NRF_PWM0->LOOP = 0; // no looping
    NRF_PWM0->DECODER = PWM_DECODER_LOAD_Individual << PWM_DECODER_LOAD_Pos;


    while (1){ 
            // read Y-axis value
            int y = getAccelerometerSample('Y');

            // map Y value to frequency range 
            uint32_t frequency = 500 + ((y + 512) * (5000 - 500)) / 1024;

            // calculate PWM counter period
            uint32_t pwm_period_ticks = 16000000 / (frequency * 16); // 16 MHz clock, prescaler DIV_16

            // set  counter reset/rollover value
            NRF_PWM0->COUNTERTOP = pwm_period_ticks;

            // start PWM sequence
            NRF_PWM0->TASKS_SEQSTART[0] = 1;
    }
    
}