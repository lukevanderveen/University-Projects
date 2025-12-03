.data

explain:    	.asciiz     	"\nOptions:\tColours:\n1 - cls\t\t1 - red\n2 - stave\t2 - orange\n3 - note\t3 - yellow\n4 - exit\t4 - green\n\t\t5 - blue\n"
input:      	.asciiz		"Select an option:"
inputC:     	.asciiz		"Select an colour:"
rows:	    	.asciiz 	"Select how many rows down you want your stave to be (between 1 and 210):\n"
notes:	    	.asciiz		"select a note\n -A\n -B\n -C\n -D\n -E\n -F\n -G\n"
errormessage:	.asciiz 	"ERROR: incorrect input\n"



black:		.word		0x00000000
red:		.word		0x00ff0000
orange:		.word		0x00ffa500
yellow:		.word		0x00ffff00
green:		.word		0x0000ff00
blue:		.word		0x000000ff
white:		.word		0x00ffffff

max:		.word 131072
base:		.word 0x10040000
counter: 	.word 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
.text

j main

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# drawLine: Procedure to draw horizontal line.

drawLine:
	lw $t4, counter
	loop3:
		sw $t3, ($t1)
		addi $t1, $t1, 4
		addi $t4, $t4, 1
		bne $t4, 512, loop3
		addi $t4, $t4, -512	#reset counter
		addi $t1, $t1, 20480	#create gap
		
		jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#drawbackground: prodcedure to draw background
drawbackground:

	addi $s0, $zero, 0
	lw $s0, base
	lw $s2, max
   	#~~~~~~~~~~~~~~~~~~~~~#
   	#display prompt for user
	addi $v0, $zero, 4
	la $a0, inputC
	syscall
   	#~~~~~~~~~~~~~~~~~~~~~#
	#read input for colours
	li $v0, 5
   	syscall
	move $t0, $v0
   	
   	#~~~~~~~~~~~~~~~~~~~~~#
   	#choose colour based on user input
   	bgt $t0, 5, error
   	blt $t0, 1, error
	beq $t0, 1, R
	beq $t0, 2, O
	beq $t0, 3, Y
	beq $t0, 4, G
	beq $t0, 5, B
	
R: #set red
	lw, $s1, red
	j loop
	
O: #set orange
	lw, $s1, orange
	j loop
	
Y: #set yellow
	lw, $s1, yellow
	j loop
	
G: #set green
	lw, $s1, green
	j loop
	
B: #set blue
	lw, $s1, blue
	j loop

   	#draw the background
	loop:
		sw $s1, 0($s0)
		addi $s0, $s0, 4
		addi $s2, $s2, -1
		bnez $s2, loop
		
		jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#compute initial gap for stave
linestart:
	
	#print string
	addi $v0,$zero,4		#put syscall service into v0
	la $a0, rows			#put address of ascii (input) into a0
	syscall				#actually print string! (this works!)
	
	#~~~~~~~~~~~~~~~~~~~~~#
#read input for options
	li $v0, 5
   	syscall
	move $t2, $v0

	bge $t2, 210, error
	ble $t2, 0, error

	li $t0, 0x10040000
	la $s7, ($t2)

	la $t1, ($t2)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t2, $t1, 2048 #length of line (512*4)

	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# note: Draw a note on the stave based on the input char
note:
	
#print string
	addi $v0,$zero,4		#put syscall service into v0
	la $a0, notes			#put address of ascii (input) into a0
	syscall				#actually print string! (this works!)
	
	addi $v0, $zero, 4
	la $a0, input
	syscall	
	
	#~~~~~~~~~~~~~~~~~~~~~#
#read input for options
	li $v0, 12
   	syscall
	move $t5, $v0

    	bgt $t5, 0x47, error
    	blt $t5, 0x41, error
	
	beq $t5, 0x41, Anote
	
	beq $t5, 0x42, Bnote
	
	beq $t5, 0x43, Cnote
	
	beq $t5, 0x44, Dnote
	
	beq $t5, 0x45, Enote
	
	beq $t5, 0x46, Fnote
	
	beq $t5, 0x47, Gnote
	
Anote:
	jal positionA
	jal playnoteA
Bnote:
	jal positionB
	jal playnoteB
Cnote:
	jal positionC
	jal playnoteC
Dnote:
	jal positionD
	jal playnoteD
Enote:
	jal positionE
	jal playnoteE
Fnote:
	jal positionF
	jal playnoteF
Gnote:
	jal positionG
	jal playnoteG
	
setnote:
	jal drawnote
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionE:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 3 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionD:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 9 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionC:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 14 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionB:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 20 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionA:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 25 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionG:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 30 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionF:
	li $t0, 0x10040000
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 36 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
drawnote:
	jal drawsmall
	jal drawsmall
	jal drawsmall
	jal drawsmall
	jal drawsmall
	jal drawsmall
	j main
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
drawsmall:
	lw $t3, black
	loop4:
		sw $t3, ($t1)
		addi $t1, $t1, 4
		addi $t4, $t4, 1
		bne $t4, 8, loop4
		addi $t4, $t4, -8	#reset counter
		addi $t1, $t1, 2016	#create gap
		
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteA:
	#an A note
	addi $v0, $zero, 33
	la $a0, 69 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteB:
	#an A note
	addi $v0, $zero, 33
	la $a0, 71 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteC:
	#an A note
	addi $v0, $zero, 33
	la $a0, 72 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteD:
	#an A note
	addi $v0, $zero, 33
	la $a0, 62 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteE:
	#an A note
	addi $v0, $zero, 33
	la $a0, 64 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteF:
	#an A note
	addi $v0, $zero, 33
	la $a0, 65 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
playnoteG:
	#an A note
	addi $v0, $zero, 33
	la $a0, 67 #pitch
	la $a1, 1000 #duration
	la $a2, 0 #instrument: piano
	la $a3, 127 #volume
	syscall
	j setnote
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
main:

	jal options
   	#~~~~~~~~~~~~~~~~~~~~~#
   	bgt $t0, 4, error
   	blt $t0, 1, error
   	beq $t0, 1, cls
   	beq $t0, 2, stave
   	beq $t0, 3, note
   	beq $t0, 4, Exit
   	#~~~~~~~~~~~~~~~~~~~~~#
   	jal options
	jal Exit
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
cls:	
    	jal drawbackground
    	j main
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
stave:
	jal linestart
	jal drawLine
	jal drawLine
	jal drawLine
	jal drawLine
	jal drawLine
	
	j main

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
options:
	#print string
	addi $v0,$zero,4		#put syscall service into v0
	la $a0, explain			#put address of string (input) into a0
	syscall				#actually print string! (this works!)
	
	addi $v0, $zero, 4
	la $a0, input
	syscall	
	
	#~~~~~~~~~~~~~~~~~~~~~#
#read input for options
	li $v0, 5
   	syscall
	move $t0, $v0
    	
	jr $ra

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
error:
	#print string
	addi $v0,$zero,4		#put syscall service into v0
	la $a0, errormessage		#put address of ascii (input) into a0
	syscall				#actually print string! (this works!)
	
	j main
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#exit program procedure
Exit:
	li $v0, 10
    	syscall
    	
    	jr $ra
