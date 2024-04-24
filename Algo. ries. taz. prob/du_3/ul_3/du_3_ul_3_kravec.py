#Autor: Marian Kravec
inp = input().strip().split()
n = int(inp[0])
k = int(inp[1])

zoz = list(map(int, input().strip().split()))

# Použijeme rolling hash s takýmito parametrami:
a = 60509
modulus=945076795243151
a_prvy = (a**(k-1)) % modulus


dic = dict()
res = ""
min_i = float("inf")

window = 0

# Vzorec ktorý použijeme je: H(x) = (x_1*a^(k-1) + ... x_k*a^0) mod modulus
for i in range(k):
    window = (window*a + zoz[i]) % modulus

for i in range(n-k+1):
    # priebezne vysledky pre hashe si zaznamename v slovniku
    # cize v pripade pythonu hash tabulke
    g = dic.get(window, "")
    if g == "":
        dic[window] = i+1
    elif g < min_i or res == "":
        res = str(g)+" "+str(i+1)
        min_i = g
    if i+k < n:
        # potom posunutie okna vieme spravit jednoducho, ze odpocitame
        # prvy clen, vynasobime a a pripocitame novy clen
        window = ((window - zoz[i]*a_prvy)*a + zoz[i+k]) % modulus

if res == "":
    print(-1)
else:
    print(res)


