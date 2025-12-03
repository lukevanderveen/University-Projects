
def RV(Px1, Px2, Px3, Py2, Py3, Py4, Py5, Pz2, Pz3, Pz4):
    
    Pa = 0.0
    Pb = 0.0
    for x in range(3):
        for y in range(4):
            Pa = x*y
            for z in range(3):
                Pb = x*y*z
                
                

    return (Pa, Pb, mean, variance)