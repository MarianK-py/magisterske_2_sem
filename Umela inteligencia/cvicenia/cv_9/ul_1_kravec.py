import matplotlib.pyplot as plt
import numpy as np

vek=np.array([50,60,37,45,80,62,34,51,49,48,26,75,72,68,72,45,49,64,72,77,54])

print("Median vekov:", np.median(vek))
print("Prvý kvantil vekov:", np.quantile(vek, 0.25))
print("Tretí kvantil vekov:", np.quantile(vek, 0.75))

print("Na boxplote vidíme, to čo nám už povedali aj hodnoty kvantilov")
print("a to, že naše dáta nie sú rovnomerne roznomerne rozdelené okolo")
print("medianu, vidíme, že 25% dát pod medianom je v intervale (48,54)")
print("čiže veľkosti 6 a 25% dát nad medianom je v intervale (54,72)")
print("čiže veľkosti 18, čiže výrazne väčšom, pri normálnom rozdelení")
print("by sme očakávali symetrické rozdelenie okolo medianu")
print("preto toto rozdelenie nemôžeme považovať za normálne!")

plt.boxplot(vek)
plt.show()


