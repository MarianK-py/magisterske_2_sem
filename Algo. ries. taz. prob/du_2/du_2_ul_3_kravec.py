# Marian Kravec

n = int(input())
m = []

for i in range(n):
    r = input().strip().split(" ")
    m.append(r)

# Pridame falosny vrchol ktory bude mat hranu hodnoty 0
# k vsetkym ostatnym vrcholom
# vdaka tomu mozeme teraz riesit ulohu
# ohodnotenej Hamiltnovskej kruznice
# kedze ide o uplny graf
# mozeme to dokonca povazovat za ulohu TSP

n += 1

print("Minimize")
line = "obj: "

for i, r in enumerate(m):
    for j, v in enumerate(r):
        if i != j:
            line += v+" x"+str(i+1)+"_"+str(j+1)+" +"

line = line.strip(" +")

print(line)

print("Subject To")

for i in range(n):
    line = "ci"+str(i+1)+": "
    for j in range(n-1):
        if i != j:
            line += "x"+str(i+1)+"_"+str(j+1)+" + "
    if i != n-1:
        line += "x"+str(i+1)+"_"+str(n)
    else:
        line = line.strip(" + ")
    line += " <= 1"
    print(line)

for i in range(n):
    line = "co"+str(i+1)+": "
    for j in range(n-1):
        if i != j:
            line += "x"+str(j+1)+"_"+str(i+1)+" + "
    if i != n-1:
        line += "x"+str(n)+"_"+str(i+1)
    else:
        line = line.strip(" + ")
    line += " <= 1"
    print(line)

for i in range(n):
    line = "cx"+str(i+1)+": "
    lineI = ""
    lineO = ""
    for j in range(n-1):
        if i != j:
            lineI += "x"+str(i+1)+"_"+str(j+1)+" + "
            lineO += "x"+str(j+1)+"_"+str(i+1)+" + "
    if i != n-1:
        lineI += "x"+str(i+1)+"_"+str(n)
        lineO += "x"+str(n)+"_"+str(i+1)
    else:
        lineI = lineI.strip(" + ")
        lineO = lineO.strip(" + ")
    line += lineI+" + "+lineO+" >= 2"
    print(line)


# potrebujeme este riesit cykly na to pouzijeme
# Miller-Tucker-Zemlin subtour elimination
# chceme aby algoritmus našiel jeden cyklus nie viac mensich
# zdroj:
# https://en.wikipedia.org/wiki/Travelling_salesman_problem#Integer_linear_programming_formulations



for i in range(1,n):
    for j in range(1,n):
        line = ""
        if i != j:
            # ui - uj + 1 <= (n-1)(1 - xi_j)
            # ui - uj + 1 <= n - 1 - n*xi_j + xi_j
            # ui - uj + 2 - n + n*xi_j - xi_j <= 0
            # ui - uj + n*xi_j - xi_j <= n - 2
            line += "cu"+str(i+1)+"_u"+str(j+1)+": u"+str(i+1)
            line += " - u"+str(j+1)+" + "+str(n)
            line += " x"+str(i+1)+"_"+str(j+1)+" - x"+str(i+1)+"_"+str(j+1)
            line += " <= "+str(n-2)
            print(line)

print("Bounds")

for i in range(1,n):
    #2 <= ui <= n
    print("2 <= u"+str(i+1)+" <= "+str(n))


print("Binary")

line = ""
for i in range(n):
    for j in range(n):
        if i != j:
            line += "x"+str(i+1)+"_"+str(j+1)+" "

print(line)

print("General")

line  = ""
for i in range(1,n):
    line += "u"+str(i+1)+" "

line.strip()
print(line)
print("End")


    
