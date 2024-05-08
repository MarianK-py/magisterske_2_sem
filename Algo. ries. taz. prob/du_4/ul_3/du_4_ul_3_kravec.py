import sys
sys.setrecursionlimit(15000)

#Autor: Marian Kravec
inp = input().strip().split()
n = int(inp[0])
k = int(inp[1])

ls = []

for i in range(n):
    ls.append(input().strip())

enum_ls = enumerate(ls)
l = len(ls[0])

ind = [0]*n

def step(indexes, val, curr_max, needed_k):
    if needed_k == 0:
        return True
    if curr_max == l:
        return False
    else:
        ok = True
        new_ind = []
        for i, v in enumerate(ls):
            if v[indexes[i]] != val:
                ok = False
        if ok:
            for i in indexes:
                new_ind.append(i+1)
            new_max = max(new_ind)
            return step(new_ind, "0", new_max, needed_k-1) or step(new_ind, "1", new_max, needed_k-1)
        else:
            for i, v in enumerate(ls):
                new_ind.append(indexes[i]+(v[indexes[i]] != val))
            new_max = max(new_ind)
            return step(new_ind, val, new_max, needed_k)

if step(ind, "0", 0, k) or step(ind, "1", 0, k):
    print("YES")
else:
    print("NO")

