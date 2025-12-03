.data
fibs: .word  0 : 12

.text

la $t0, fibs

addi $t1, $zero, 0
sw $t1, 0($t0)
addi $t1, $zero, 1
sw $t1, 4($t0)

addi $t2, $zero, 2
loop: 	beq $t2, 12, end

	sll $t4, $t2, 2
	add $t4, $t0, $t4
	
	lw $t5, -8($t4)
	lw $t6, -4($t4)
	add $t5, $t5, $t6
	sw $t5, 0($t4)

	addi $t2, $t2, 1
	j loop
end:    nop#