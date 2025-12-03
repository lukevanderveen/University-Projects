import hashlib
# Task 1

#1.1
# Include any required modules
def HexBinSHA256(stringToConvert):
    h = hashlib.sha256()
    h.update(stringToConvert)
    hexValue = h.hexdigest()
    tempInt = int(hexValue, 16)
    binValue = bin(tempInt)
    return (hexValue, binValue)

#1.2
def bruteSHA256(shaString):
    #TODO
    # establish alphabet
    # go through every possible string until you find match to the hexdigest
    
    return password
    

# Main
if __name__ == '__main__':
    #1.1
    s1 = b"Hello World"
    hex, bin = HexBinSHA256(s1)
    print(s1)
    print(hex)
    print(bin)
    
    #1.2
    sha = "94f94c9c97bfa92bd267f70e2abd266b069428c282f30ad521d486a069918925"
    

