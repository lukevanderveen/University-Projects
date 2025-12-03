#include "MicroBit.h"
#include <iostream>

extern NRF52Serial serial;   // serial object may be useful for debugging 
//font: monocraft
//###################################################################################
// Write your code for Subtask 1 here (and leave these separators in your code!):
/*
commands: move MICROBIT.hex D:\
*/
//
void printBinary(uint8_t value) {
    serial.printf("0b"); 
    for (int i = 7; i >= 0; i--) { 
        serial.printf("%d", (value >> i) & 1); 
    }
}

void printPinConfig(uint8_t pin) {
    if (pin >= 32) {
        serial.printf("Invalid pin number\n");
        return;
    }
    
    serial.printf("Pin %d configuration register state: 0b", pin);
    for (int i = 31; i >= 0; i--) {
        serial.printf("%d", (NRF_P0->PIN_CNF[pin] >> i) & 1);
    }
    serial.printf("\n");
}

void printOutRegister() {
    serial.printf("OUT register state: 0b");
    for (int i = 31; i >= 0; i--) {
        serial.printf("%d", (NRF_P0->OUT >> i) & 1);
    }
    serial.printf("\n");
}



void displayBinary(uint8_t value) {
    //serial.printf("displayBinary: %d\n", value);
    static bool initalised = false;
    if (!initalised) {

        // configure Row 1 (P0.21) and Columns (P0.28, P0.11, P0.31, P1.05, P0.30) as outputs
        NRF_P0->PIN_CNF[21] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
                            
        NRF_P0->PIN_CNF[28] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
        
        NRF_P0->PIN_CNF[11] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
        
        NRF_P0->PIN_CNF[31] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
                            
        NRF_P0->PIN_CNF[30] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);
                            
        NRF_P1->PIN_CNF[5] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                            (GPIO_PIN_CNF_INPUT_Disconnect << GPIO_PIN_CNF_INPUT_Pos) |
                            (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                            (GPIO_PIN_CNF_DRIVE_H0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                            (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);

        NRF_P0->DIR |= (1 << 21) | (1 << 28) | (1 << 11) | (1 << 31) | (1 << 30);
        NRF_P1->DIR |= (1 << 5);

        initalised = true;
        serial.printf("initalised\n");
    }   
    
    int cols[5] = {28, 11, 31, 5, 30};

    uint8_t ledRow = value & 0b11111;
    serial.printf("ledRow: %d (binary: ", ledRow);
    printBinary(ledRow); 
    serial.printf(")\n"); 
    serial.printf("\n");

    ledRow = ((ledRow & 0b00001) << 4) | 
              ((ledRow & 0b00010) << 2) | 
              ((ledRow & 0b00100) << 0) | 
              ((ledRow & 0b01000) >> 2) | 
              ((ledRow & 0b10000) >> 4);

    NRF_P0->OUT |= (1 << 21);  
    for (int i = 0; i < 5; i++) {
        if (ledRow & (1 << i)) {
            if (cols[i] == 5){
                NRF_P1->OUT &= ~(1 << (cols[i])); 
                //serial.printf("Column 4 set low\n");
                //serial.printf("\n");
            } else {
                NRF_P0->OUT &= ~(1 << (cols[i])); 
                //serial.printf("Column %d set low\n", i + 1);
                //serial.printf("\n");
            }
        } else {
            if (cols[i] == 5){
                NRF_P1->OUT |= (1 << (cols[i])); 
                //serial.printf("Column 4 set high\n");
                //serial.printf("\n");
            } else {
                NRF_P0->OUT |= (1 << (cols[i])); 
                //serial.printf("Column %d set high\n", i + 1);
                //serial.printf("\n");
            }
        }

        //printOutRegister();
        //serial.printf("\n");
    }
}

//
void countUpBinary(uint8_t initialValue) {
    uint8_t counter = initialValue & 0b11111;  

    while (true) {
        displayBinary(counter);  

        for (volatile int i = 0; i < 1470000; i++);

        if (counter == 0b11111) {
            counter = 0b00000;
            break;  
        }

        counter = (counter + 1) & 0b11111; 

        serial.printf("Counter incremented to: %d (binary: ", counter);
        printBinary(counter);
        serial.printf(")\n");
    }

    
    displayBinary(counter);  
    serial.printf("Counter reset to: %d (binary: ", counter);
    printBinary(counter);
    serial.printf(")\n");
}

//###################################################################################
// Write your additional code for Subtask 2 here:

void buttonConfiguration(){
    
    NRF_P0->PIN_CNF[14] = (GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos) |
                          (GPIO_PIN_CNF_INPUT_Connect << GPIO_PIN_CNF_INPUT_Pos) |
                          (GPIO_PIN_CNF_PULL_Pullup << GPIO_PIN_CNF_PULL_Pos) |
                          (GPIO_PIN_CNF_DRIVE_S0H1 << GPIO_PIN_CNF_DRIVE_Pos) | 
                          (GPIO_PIN_CNF_SENSE_High << GPIO_PIN_CNF_SENSE_Pos); 

    printPinConfig(14);

    
    NRF_P0->PIN_CNF[23] = (GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos) |
                          (GPIO_PIN_CNF_INPUT_Connect << GPIO_PIN_CNF_INPUT_Pos) |
                          (GPIO_PIN_CNF_PULL_Pullup << GPIO_PIN_CNF_PULL_Pos) |
                          (GPIO_PIN_CNF_DRIVE_S0H1 << GPIO_PIN_CNF_DRIVE_Pos) | 
                          (GPIO_PIN_CNF_SENSE_High << GPIO_PIN_CNF_SENSE_Pos); 

    printPinConfig(23);

    //NRF_P0->DIR |= (1 << 14) | (1 << 23);
}

bool isButtonAPressed() {
    return !(NRF_P0->IN & (1 << 14));  
}


bool isButtonBPressed() {
    return !(NRF_P0->IN & (1 << 23));  
}


//
void countWithButtonsBinary(uint8_t initialValue) {
    serial.printf(" ... Subtask 2 running ... \n"); 
    buttonConfiguration();

    uint8_t counter = initialValue & 0b11111;  
    displayBinary(counter);                    
    static bool prevA = false, prevB = false;    


    while (1){


        bool currentA = isButtonAPressed();
        bool currentB = isButtonBPressed();

        if (currentA && !prevA ) {
            if (counter == 0){
                counter = 0b11111;
            }else{
                counter -= 1;
            }
            displayBinary(counter);                
            serial.printf("Button A pressed, counter: %d\n", counter);
        prevA = true;
        }else if(!currentA && prevA){
            prevA = false;
        }

        if (currentB && !prevB ) {
            //for (volatile int i = 0; i < 50000; i++);
            if (counter == 0b11111){
                counter = 0;
            }else{
                counter += 1;
            }
            displayBinary(counter);
        prevB = true;
        }else if(!currentB && prevB){
            prevB = false;
        }
        for (volatile int i = 0; i < 500000; i++);
    }
}

//###################################################################################
// Write your additional code for Subtask 3 here:

static int16_t adc_buffer = 0;

void configureADCForP0() {
    serial.printf("Configuring ADC for P0...\n");
   NRF_P0->PIN_CNF[0] = (GPIO_PIN_CNF_DIR_Input << GPIO_PIN_CNF_DIR_Pos) |
                        (GPIO_PIN_CNF_INPUT_Connect << GPIO_PIN_CNF_INPUT_Pos) |
                        (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) |
                        (GPIO_PIN_CNF_DRIVE_S0H1 << GPIO_PIN_CNF_DRIVE_Pos) |
                        (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos);

    // Enable the SAADC
    NRF_SAADC->ENABLE = (SAADC_ENABLE_ENABLE_Enabled << SAADC_ENABLE_ENABLE_Pos);
    //serial.printf("SAADC enabled.\n");
    
    NRF_SAADC->RESOLUTION = (SAADC_RESOLUTION_VAL_10bit << SAADC_RESOLUTION_VAL_Pos); 
    NRF_SAADC->OVERSAMPLE = (SAADC_OVERSAMPLE_OVERSAMPLE_Bypass << SAADC_OVERSAMPLE_OVERSAMPLE_Pos);
 
    // Configure channel 0 for P0
    NRF_SAADC->CH[0].PSELP = SAADC_CH_PSELP_PSELP_AnalogInput0; 
    NRF_SAADC->CH[0].CONFIG = (SAADC_CH_CONFIG_GAIN_Gain1 << SAADC_CH_CONFIG_GAIN_Pos) | 
                              (SAADC_CH_CONFIG_REFSEL_VDD1_4 << SAADC_CH_CONFIG_REFSEL_Pos) | 
                              (SAADC_CH_CONFIG_TACQ_10us << SAADC_CH_CONFIG_TACQ_Pos) | 
                              (SAADC_CH_CONFIG_MODE_SE << SAADC_CH_CONFIG_MODE_Pos);
    //serial.printf("ADC configured for P0 with Gain1 and VDD/4 reference.\n");

    NRF_SAADC->RESULT.PTR = (uint32_t)&adc_buffer;
    NRF_SAADC->RESULT.MAXCNT = 1; // <---------------------
    //serial.printf("buffer updated\n");
}

//
uint8_t sampleVoltage() {
    //serial.printf("Starting to read analog voltage on P0...\n");
    NRF_SAADC->EVENTS_STARTED = 0;
    NRF_SAADC->EVENTS_END = 0;
    NRF_SAADC->EVENTS_STOPPED = 0;
    configureADCForP0();

    NRF_SAADC->TASKS_START = 1;
    while (!NRF_SAADC->EVENTS_STARTED);
    NRF_SAADC->EVENTS_STARTED = 0;
    //serial.printf("ADC started.\n");    

    NRF_SAADC->TASKS_SAMPLE = 1;
    while (!NRF_SAADC->EVENTS_END);
    NRF_SAADC->EVENTS_END = 0;
    //serial.printf("Sampling complete.\n");

    int8_t adcVal = (adc_buffer * 255) / 1023; 
    //serial.printf("ADC result read: %d\n", adcVal);

    NRF_SAADC->TASKS_STOP = 1;
    while (!NRF_SAADC->EVENTS_STOPPED);
    NRF_SAADC->ENABLE = (SAADC_ENABLE_ENABLE_Disabled << SAADC_ENABLE_ENABLE_Pos);
    //serial.printf("SAADC stopped and disabled.\n");

    return adcVal;
}

//
void displayVoltageBinary() {
    serial.printf(" ... Subtask 3 running ... \n"); 
    while (1){
        uint8_t sample = sampleVoltage();
        for(int i = 0; i < 50000;i++);
        displayBinary(sample);
    }  // NB this function never returns
    
}

//###################################################################################
// Write your additional code for Subtask 4 here:

//
void driveRGB() {
    serial.printf(" ... Subtask 4 running ... \n"); 
    while (1);  // NB this function never returns
}

//###################################################################################
// Write your additional code for Subtask 5 (stretch goal) here:

// 
void countWithTouchesBinary(uint8_t initialValue) {
    serial.printf(" ... Subtask 5 running ... \n"); 
    while (1);  // NB this function never returns
}