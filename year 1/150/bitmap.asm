.data
max:	.word 131072
base:	.word 0x10010000
count:	.word 100
#jmp:	.word 100
.text
	addi $s0, $zero, 0
	lw $s0, base
	li $s1, 0x00ff0000
	lw $s2, max
	
#background
loop:
	sw $s1, 0($s0)
	addi $s0, $s0, 4
	addi $s2, $s2, -1
	bnez $s2, loop

#line, horizontal
	lw $s3, count
	li $s4, 0x0000ff00
	lw, $s5, base
#	lw $s6, jmp
	
#locate:
#	addi $s5, $s5, 4
#	addi $s6, $s6, -1
#	bnez $s6, locate 
	
loop1:
	sw $s4, 0($s5)
	addi $s5, $s5, 4
	addi $s3, $s3, -1
	bnez $s3, loop1
	
