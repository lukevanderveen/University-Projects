.data
max:		.word 131072
base:		.word 0x10040000
counter: 	.word 0
counter2:
white:		.word 0x00FFFFFF
rows:	    .asciiz 	"Select how many rows down you want your stave to be (between 1 and 210):\n"
input:      .asciiz	"Select an option:\n"
errormessage:	    .asciiz 	"ERROR: incorrect input"
.text
j main
main:
	addi $t0, $zero, 0
	lw $t1, base
	lw $s2, max
	
	
	lw $t4, counter#counts length
	#lw $t5, counter2#counts width
	
	jal linestart
	jal drawLine
	jal drawLine
	jal drawLine
	jal drawLine
	jal drawLine
	
	jal positionA
	jal drawnote
	
	li $v0, 10
    	syscall

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
drawsmall:
	lw $t3, white
	loop2:
		sw $t3, ($t1)
		addi $t1, $t1, 4
		addi $t4, $t4, 1
		bne $t4, 8, loop2
		addi $t4, $t4, -8	#reset counter
		addi $t1, $t1, 2016	#create gap
		
		jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
drawnote:
	jal drawsmall
	jal drawsmall
	jal drawsmall
	jal drawsmall
	jal drawsmall
	jal drawsmall
	
	jr $ra

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	

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
	
#print to test 
	li $v0, 1
    	move $a0, $t2
    	syscall

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
positionE:
	
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 3 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionD:
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 9 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionC:
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 14 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionB:
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 20 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionA:
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 25 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionG:
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 30 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
positionF:
	la $t1, ($s7) #rember where stave is
	addi $t1, $t1, 36 #where on Y axis (4 bits)
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 1024 #where on x (4 bits)
	
	jr $ra
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# drawLine: Procedure to draw horizontal line.

drawLine:
	lw $t4, counter
	lw $t3, white
	loop3:
		sw $t3, ($t1)
		addi $t1, $t1, 4
		addi $t4, $t4, 1
		bne $t4, 512, loop3
		addi $t4, $t4, -512	#reset counter
		addi $t1, $t1, 20480	#create gap
		
		jr $ra
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
error:
	#print string
	addi $v0,$zero,4		#put syscall service into v0
	la $a0, errormessage		#put address of ascii (input) into a0
	syscall				#actually print string! (this works!)
	
	li $v0, 10
    	syscall
	
#unsused code
	#an A note
	#addi $v0, $zero, 33
	#la $a0, 69 #pitch
	#la $a1, 1000 #duration
	#la $a2, 0 #instrument: piano
	#la $a3, 40 #volume
	#syscall


#li $t1, 60
	#sll $t1, $t1, 11
	#add $t1, $t1, $t0
	#addi $t2, $t1, 2048 #length of line (512*4)

	#li $t3, 0x00000000 # colour of line
	
	#jal drawbackground
	#jal drawLine
	#jal drawLine
	#jal drawLine
	#jal drawLine
	#jal drawLine

	#li $v0, 10
	#syscall
