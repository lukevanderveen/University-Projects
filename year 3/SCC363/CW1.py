# INCLUDE MODULES
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def XOR_bytes(b1: bytes, b2: bytes) -> bytes:
    # XOR function 
    # use the XOR zipping method where for a byte in a, XOR it with its corresponding byte in b
    return bytes(a ^ b for a, b in zip(b1, b2))

'''
:param key: str: The hexadecimal value of a key to be used for encryption
:param iv: str: The hexadecimal value of an initialisation vector to be
used for encryption
:param data: str: The data to be encrypted
:return: str: The hexadecimal value of encrypted data
'''
def Encrypt(key: str, iv: str, data: str) -> str:
    # convert key and IV from hex to bytes 
    # convert data to bytes with .encode() 
    key = bytes.fromhex(key)
    iv = bytes.fromhex(iv)
    data = data.encode('utf-8') 
    
    # a block of data in AES is limited to 16 bytes so we set a block_size of 16 as a limiter
    # initalise a list to store the blocks
    # if the length of the data is less than or equal to block size just add data to the blocks list
    # if it is greater, split th data into 16 byte sized blocks and append them to the list
    block_size = 16    
    blocks = []
    if len(data) <= block_size:
        blocks = [data]
    else:
        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]
            blocks.append(block)
    
    # create a cipher text storage list for each individual block
    # initalise current IV as IV (this will change as we use each cipher text as the next IV)        
    ct = []
    current_iv = iv
    
    # For each block in the list of blocks
    # Encrypt current IV with AES key in ECB mode as an encryptor
    # XOR the result with plaintext the plaintext to create the ciphertext
    # Add result to the ciphertext list and use the result as the next IV
    for block in blocks:
        encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        ct_i = encryptor.update(current_iv) + encryptor.finalize()
        ct_i = XOR_bytes(block, ct_i)
        ct.append(ct_i.hex())
        current_iv = ct_i
        
    # return the ciphertext as a comma seperated hex string
    return ','.join(ct)


'''
:param key: str: The hexadecimal value of a key to be used for decryption
:param iv: str: The hexadecimal value of the initialisation vector to be
used for decryption
:param data: str: The hexadecimal value of the data to be decrypted
:return: str: The decrypted data in UTF-8 format
'''
def Decrypt(key: str, iv: str, data: str) -> str:
    # convert key and IV from hex to bytes 
    key = bytes.fromhex(key)
    iv = bytes.fromhex(iv)
    

    # split ciphertext into individual blocks
    # use a comma as a seperator to symbolise a seperate block
    # create a list for plain text blocks
    # initalise current IV as IV (this will change as we use each cipher text as the next IV)        
    ct_blocks = [bytes.fromhex(block) for block in data.split(",")]
    pt_blocks = []
    current_iv = iv
    
    # process the ciphertexts block by block
    # convert the ciphertext to bytes from hex
    # initalise a decryptor object in ECB mode  with the AES key
    # create plaintext using the ciphertext
    # append the plaintext to the list of data
    # make the current IV the ciphertext in order to decrypt the next block
    for ct_block in ct_blocks:
        #ct = bytes.fromhex(ct_block)
        decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
        pt = decryptor.update(current_iv) + decryptor.finalize()
        pt = XOR_bytes(ct_block, pt)
        pt_blocks.append(pt)
        current_iv = ct_block
        
    #combine plain texts and then return the string converted from bytes to regular string  
    combined_pt = b''.join(pt_blocks)
    
    return combined_pt.decode('utf-8')
    
    
    # -- END OF YOUR CODERUNNER SUBMISSION CODE
    # You can test your code in your system (NOT IN YOUR CODERUNNER SUBMISSION)

# Main
if __name__ == "__main__":
    # Task 1
    key = "2b7e151628aed2a6abf7158809cf4f3c"
    iv = "000102030405060708090a0b0c0d0e0f"
    text = "Hello World"

    ct = Encrypt(key, iv, text)
    pt = Decrypt(key, iv, ct)
    print(ct)
    print(pt)

'''
TEST CASE OUTPUT:
189b0ba0f64d65d9a86553
Hello World
'''