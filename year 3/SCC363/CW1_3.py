# -- START OF YOUR CODERUNNER SUBMISSION CODE
# INCLUDE MODULES
import string
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import random
# INCLUDE HELPER FUNCTIONS YOU IMPLEMENT

'''
param: data: bytes: The data to be hashed
return: bytes: The truncated hash
'''
def myHash(data: bytes) -> bytes:
    # intialise a hash object in SHA256 mode
    # update the hash with the data input
    # create a final SHA256 object  
    h = hashes.Hash(hashes.SHA256(), backend=default_backend)
    h.update(data)
    hash_complete = h.finalize()
    
    # split the hash in 2 
    # XOR the 2 values together 
    # use the XOR zipping method where for a byte in hash half a, XOR it with its corresponding byte in hash half b
    # repeat this 3 times
    for _ in range(3):
        a = hash_complete[:len(hash_complete)//2]
        b = hash_complete[len(hash_complete)//2:]
        
        hash_complete = bytes(a ^ b for a, b in zip(a, b))

    return hash_complete


'''
return: str: Return YES if myHash is secure and NO otherwise

Attack: Birthday attack, 
A birthday attack is used to detect collisions in a hash functionit works off the principle that 
given enough attempts 2 different strings will produce the same hash, if they do not then a hash
function can be considered secure. SHA256 produces a 32 byte output hash, but the repeated XORing
and reshaping has reduced its size create a weakness in being a shorter length
hash
'''
def myAttack() -> str:
    # create a dictionary to store previouslt generated hash values
    # define a limit to the number of attacks
    checked = {}
    trial_limit = 10000
    
    # generate a 6 character string populated by random ascii letters and digits
    # convert the string to bytes
    # use the custom hash function called myHash() on the string that has been generated
    for _ in range(trial_limit):
        st = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        data = st.encode()
        h = myHash(data)
        
        # if the generated hash is in the dictionary of previouslt generated hashes
        # check whether the collision occurs with different inputs, if it does return "NO" it is not a secure hash funtion
        # otherwise store the new hash value in checked
        if h in checked:
            if checked[h] != data:
                return "NO"
        else:
            checked[h] = data   
    
    # if there were no collisions within the trial limit
    # return "YES" it is a secure hash function
    return "YES"


if __name__ == "__main__":
    print(myHash(b"a"))
    print(myAttack())
    '''
    TEST CASE OUTPUT FOR myHash():
    print(myHash(b”a”)
    b'\xc5\xf9O\x92'
    '''