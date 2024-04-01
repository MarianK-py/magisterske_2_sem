# Autor: Marian Kravec

n = int(input())
graph = {}
for i in range(n):
    graph[str(i+1)] = []
for i in range((n*(n-1))//2):
    inp = input().split(" ")
    graph[inp[0]].append(inp[1])


# algoritmus hlada cykly dlzky 3 a odstrani vsetky tri vrcholy cyklu
# vdaka uplnosti grafu staci odstranit cykly dlzky 3 na odstranenie vsetkych cyklov
# dovod: kazdy cyklus dlzky n obsahuje aspon jeden vrchol z nejakeho cykla dlzky 3
# (dokonca vieme ukazat, ze kazdy vrchol tohto cyklu patri nejakemu cyklu dlzky 3)
# ukazeme si to na priklade cyklu dlzky 4
# majme takyto cyklus: A->B->C->D->A
# kedze mame uplny graf medzi vrcholom A a C existuje hrana
# ak je tam hrana A->C tak v grafe existuje cyklus A->C->D->A co je cyklus dlzky 3
# ak je tam hrana C->A tak v grafe existuje cyklus A->B->C->A co je cyklus dlzky 3
# cize ak odstranime vsekty cykly dlzky tym ze odstranime vsetky ich vrcholy
# odstranime aj vsetky dlhsie cykly z grafu
# kedze vsak teoreticky by mohlo stacit odstranit aj iba jeden z troch vrcholov cyklu 3
# (avsak aspon jeden urcite) tak je nase riesenie najhorie 3-APX


# pouzijeme v podstate DFS s ohranicenim na maximalnu hlbku 3
def find_cycles_of_length_3(graph):
    # pamatame si vrcholy ktore odstranime
    to_remove = set()
    # pre kazdy vrchol hladame cykly
    for vertex in list(graph.keys()):
        # ak ho chceme odstranit (lebo uz bol v inom cykle) preskocime ho
        if vertex not in to_remove:
            # pamatame si vrcholy ktore sme uz skontrolovali
            visited = set()
            # neprehladane cesty si zapamatavame v stacku
            stack = [(vertex, [vertex])]
            # hladame kym nevyprazdnime stack
            while stack:
                current, path = stack.pop()
                visited.add(current)
                # ak je cesta uz dlzky 4 zahodime ju
                if len(path) < 4:
                    # skusime sa poshnut do kazdeho zo susedov
                    for neighbor in graph[current]:
                        # ak suseda chceme odstranit preskocime ho
                        if neighbor not in to_remove:
                            # ak nam vznikol cyklus a je dlzky 3 tak vsetky
                            # vrcholy cyklu dame do množiny na vyhodenie
                            if neighbor in path and len(path) == 3: 
                                to_remove.add(path[0])
                                to_remove.add(path[1])
                                to_remove.add(path[2])
                            # ak sme vrchol este nevideli tak ho pridame do stacku
                            elif neighbor not in visited:
                                stack.append((neighbor, path + [neighbor]))
    return list(to_remove)

# Zlozitost by malo byt ~O(n^3) kedze hladame cykly dlzky 3

to_remove = find_cycles_of_length_3(graph)

print(len(to_remove))
print(" ".join(to_remove))
