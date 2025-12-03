# Include any required modules
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt(key, plainText):
    # Generate an initialisation vector
    iv = os.urandom(16)
    # Construct an AES-CTR Cipher object with the given key and
    encryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
    # randomly generated initialisation vector
    cipherText = encryptor.update(plainText) + encryptor.finalize()
    return (cipherText, iv)

def decrypt(key, cipherText, iv):
    # Construct an AES-CTR Cipher object with the given key and
    decryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).decryptor()
    # iv to decrypt
    plainText = decryptor.update(cipherText) + decryptor.finalize()

    return (plainText)

def myPKCS7(data, block_size):
    l = len(data)
    padLength = block_size - (l%block_size)
    padded_data = bytearray(data)
    
    padded_data = padded_data.extend(bytes([padLength] * padLength))
    return(padded_data)

# Main
if __name__ == '__main__':
    """
    1. Create a key for AES and a plaintext
    2. Encrypt the plaintext and print the result
    3. Decrypt the ciphertext and print the result
    """
    key = os.urandom(16)
    plainText = b'AbCdEfGh'
    
    cipherText, iv = encrypt(key, plainText)
    finalPlainText = decrypt(key, cipherText, iv)
    
    print(plainText)
    print(cipherText)
    print(finalPlainText)
    
    print(myPKCS7(b'a'*13, 16))
    
    
    # INCLUDE MODULES
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

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
    # Encrypt current IV with AES key in CTR mode (which uses XOR to encrypt data) as an encryptor
    # XOR the result with plaintext the plaintext to create the ciphertext
    # Add result to the ciphertext list and use the result as the next IV
    for block in blocks:
        encryptor = Cipher(algorithms.AES(key), modes.CTR(current_iv)).encryptor()
        ct_i = encryptor.update(block) + encryptor.finalize()
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
    ct_blocks = data.split(',')
    pt_blocks = []
    current_iv = iv
    
    # process the ciphertexts block by block
    # convert the ciphertext to bytes from hex
    # initalise a decryptor object in CTR mode  with the AES key
    # create plaintext using the ciphertext
    # append the plaintext to the list of data
    # make the current IV the ciphertext in order to decrypt the next block
    for ct_block in ct_blocks:
        ct = bytes.fromhex(ct_block)
        decryptor = Cipher(algorithms.AES(key), modes.CTR(current_iv)).decryptor()
        pt = decryptor.update(ct) + decryptor.finalize()
        pt_blocks.append(pt)
        current_iv = ct
        
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