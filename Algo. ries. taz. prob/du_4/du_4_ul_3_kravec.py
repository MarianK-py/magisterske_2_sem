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
        new_ind = []
        # kazdy index posunie za poziciu kde maju vsetky stringy rovnaku hodnotu val
        for i, v in enumerate(ls):
            pos = indexes[i]
            if v[pos] != val:
                new_pos = pos+1
                while new_pos < l and v[new_pos] != val:
                    new_pos += 1
                if new_pos == l:
                    return False
                new_ind.append(new_pos+1)
            else:
                if pos == l:
                    return False
                new_ind.append(pos+1)
        new_max = max(new_ind)
        # hlada spolocne podmnoziny o jedno mensie zacinajuce jednym alebo druhym znakom, z od novych indexov dalej
        return step(new_ind, "0", new_max, needed_k-1) or step(new_ind, "1", new_max, needed_k-1)

if step(ind, "0", 0, k) or step(ind, "1", 0, k):
    print("YES")
else:
    print("NO")

