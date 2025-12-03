.data
max:		.word 131072
base:		.word 0x10040000
counter: 	.word 0
counterHoriz:	.word 0
white:		.word 0x00FFFFFF
.text
j main
main:
	addi $t0, $zero, 0
	lw $s2, max
	
	lw $t0, base
	lw $t3, white
	
	#positions horizontals
	li $t1, 30 #where on Y axis
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 40 #on x axis
	
	addi $t2, $t1, 32
	#horizontal line
	loop3:
		sw $t3, ($t1)
		sw $t3, 12288($t1)#tells it to how many lines next horizontal should be on
		addi $t1, $t1, 4
		bne $t1, $t2, loop3
		
	#positions verticals	
	li $t1, 30 #where on y axis
	sll $t1, $t1, 11
	add $t1, $t1, $t0
	addi $t1, $t1, 40 #x axis
	
	addi $t2, $t1, 14336#vertical length
	#vertical line
	loop4:
		sw $t3, ($t1)
		sw $t3, 32($t1)#vertical line width spacing
		addi $t1, $t1, 2048#go next line
		bne $t1, $t2, loop4
	
	
	
	li $v0, 10
    	syscall

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#	
#unsused code
	#an A note
	#addi $v0, $zero, 31
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
	#jal drawline
	#jal drawline
	#jal drawline
	#jal drawline
	#jal drawline

	#li $v0, 10
	#syscall
