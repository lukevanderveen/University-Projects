# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE MODULES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
# INCLUDE HELPER FUNCTIONS YOU IMPLEMENT

def pad_pkcs7(data: bytes, block_size: int = 16) -> bytes:
    """
    Applies PKCS7 padding to a given byte string.
    :param data: Input data in bytes.
    :param block_size: Block size (default: 16 bytes for AES).
    :return: Padded data.
    """
    padder = padding.PKCS7(block_size * 8).padder()
    return padder.update(data) + padder.finalize()

'''
:param plaintext1: str: This is Plaintext1
:param ciphertext1: bytes: This is the ciphertext of Plaintext1
:param plaintext2: str: This is Plaintext2
:return: bytes: The ciphertext of Plaintext2
'''

"""
attack: known plaintext attack, time time pad attack, key stream reuse
here we take advantage of CTR mode's keystream reuse, as we don't have the key or the IV, 
but we know that ciphertext created is used as the new IV in a traditional block by block
AES CTR, so we create a new keystream (IV) by XORing the plaintext with the ciphertext.
We then use this as our new key and XOR it with the plaintext2.
As the first few bytes of the of the plaintexts are exactly the same they should 
produce the same ciphertext for those corresponding bytes
plaintext matches: This is your General. AND This is your General. 
ciphertext matches: \xf2\x0f\x97#$D\xa8\xda\xa0\xe4`:TQ\x82%\xc3\x15\x9f<*\r\ AND 
                    \xf2\x0f\x97#$D\xa8\xda\xa0\xe4`:TQ\x82%\xc3\x15\x9f<*\r\

"""
def attackAESMode(plaintext1: str, ciphertext1: bytes, plaintext2: str) -> bytes:
    # convert the plaintexts to bytes using .encode()
    pt1 = plaintext1.encode('utf-8')
    pt2 = plaintext2.encode('utf-8')
    
    # recover the keystream by XORing the bytes from plaintext1 and ciphertext1
    # use the zipping method where each byte from plaintext1 is XORed with the corresponding byte from ciphertext1 and then zipped togehter
    # use the keystream to XOR with plaintext2 in order to generate ciphertext2 
    # then return ciphertext 2
    # Ensure plaintext1 and ciphertext1 have the same length
    
    keystream = bytes(a ^ b for a, b in zip(pt1, ciphertext1))
    ciphertext2 = bytes(a ^ b for a, b in zip(pt2, keystream))
    
    return ciphertext2


# Main
if __name__ == "__main__":
    pt1 = "This is your General. Hold position until further orders. I repeat, hold position."
    ct1 = b'\xf2\x0f\x97#$D\xa8\xda\xa0\xe4`:TQ\x82%\xc3\x15\x9f<*\r\x93\x95\xb5\xef58\x1be\x8e?\xcf\x08\x90pqC\xaf\x93\xb5\xabs=\x06b\x8f.\xd4G\x91"`H\xa9\x89\
            xf7\xab\\h\x06s\x97.\xc7\x13\xd2plB\xb7\x9e\xf9\xfbz;\x1db\x8e$\xc8I'
   
    pt2 = "This is your General. Proceed with the attack at dawn. I repeat, proceed with the attack at dawn."
    print(attackAESMode(pt1, ct1, pt2))
