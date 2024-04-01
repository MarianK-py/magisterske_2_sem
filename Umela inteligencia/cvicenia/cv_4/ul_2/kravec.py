import random
import numpy

# Vypracoval: Marian Kravec

def naHriadely(x, y):
    if x < -80:
        return False
    elif x < -75:
        xTemp = x+80
        yTemp = 20+xTemp
        return y < yTemp
    elif x < -40:
        return y < 20
    elif x < -30:
        return y < 30
    elif x < -10:
        return y < 20
    elif x < 10:
        return y < (10*(0.5 - 0.005*(x**2))+20)
    elif x < 20:
        return y < 20
    elif x < 30:
        return y < 40
    elif x < 60:
        return y < 25
    elif x < 65:
        xTemp = x-60
        yTemp = 25-xTemp
        return y < yTemp
    else:
        return False

def sampler(l, w, n, m):
    lPol = l/2
    wPol = w/2
    objCeleho = l*(w**2)
    objValca = l*(wPol**2)*numpy.pi
    res = []
    for i in range(n):
        pocet = 0
        for j in range(m):
            x = lPol*(2*random.random()-1)
            y = wPol*(2*random.random()-1)
            z = wPol*(2*random.random()-1)
            y2D = (y**2 + z**2)**(1/2)
            #print(y2D)
            pocet += naHriadely(x, y2D)
        castObj = pocet/m
        res.append(castObj*objCeleho)
    objHriadel = numpy.mean(res)
    objOdpad = objValca - objHriadel
    return objHriadel, objOdpad

h, o = sampler(150, 85, 10, 10000)
print(f"Hriadel ma objem {h:.3f} mm3")
print(f"Pri sustruzeni vznikne {o:.3f} mm3 odpadu")

def __main__():
    h, o = sampler(150, 85, 10, 10000)
    print(f"Hriadel ma objem {h:.3f} mm3")
    print(f"Pri sustruzeni vznikne {o:.3f} mm3 odpadu")
    return