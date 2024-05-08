from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

gen_model = hmm.CategoricalHMM(n_components=3, random_state=42)

# modelujeme semafor, a pozorujeme či autá prechádzajú cez križovatku

# na začiatok je v stave 0 čo je červená
gen_model.startprob_ = np.array([1.0, 0.0, 0.0])

# prechody sú následovne (z cervenej nikdy neprejde hned na zelenu a naopak)
# (zaroven ide o zvlastny semafor ktory vie prejst z cervenej do oranzovej a spat do cervenej)
gen_model.transmat_ = np.array([[0.7, 0.3, 0],  # cervena -> ...
                                [0.3, 0.4, 0.3],# oranzova -> ...
                                [0, 0.2, 0.8]]) # zelena -> ...


gen_model.emissionprob_ = np.array([[0.95, 0.05], # na cervenu ide malo aut
                                    [0.3, 0.7],   # na oranzovu uz viac
                                    [0.01, 0.99]])# na zelenu takmer vsetci


pozorovania, gen_states = gen_model.sample(10000)

total_score = gen_model.score(pozorovania)

#np.savetxt('pozorovania.txt', pozorovania)
#np.savetxt('orig_stavy.txt', gen_states)

pozorovania = np.array([[xi] for xi in np.loadtxt('pozorovania_viky.txt',dtype=np.int32)])
gen_states = np.array([[xi] for xi in np.loadtxt('orig_stavy_viky.txt',dtype=np.int32)])

X_train = pozorovania[:9*(pozorovania.shape[0] // 10)]
X_validate = pozorovania[9*(pozorovania.shape[0] // 10):]

gen_score = gen_model.score(X_validate)

best_score = best_model = None
n_fits = 50
np.random.seed(54)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(
        n_components=3, random_state=idx,
        init_params='se')
    model.transmat_ = np.array([np.random.dirichlet([0.4, 0.35, 0.25]),
                                np.random.dirichlet([0.3, 0.4, 0.3]),
                                np.random.dirichlet([0.25, 0.35, 0.4])])
    model.fit(X_train)
    score = model.score(X_validate)
    print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score

print(f'Generated score: {gen_score}\nBest score:      {best_score}')

# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = best_model.predict(pozorovania)

assumed_score = best_model.score(pozorovania)

print(np.sum(states == gen_states))

print(f'Transmission Matrix Generated:\n{gen_model.transmat_.round(3)}\n\n'
      f'Transmission Matrix Recovered:\n{best_model.transmat_.round(3)}\n\n')

print(f'Emision Matrix Generated:\n{gen_model.emissionprob_.round(3)}\n\n'
      f'Emision Matrix Recovered:\n{best_model.emissionprob_.round(3)}\n\n')

print(f"Absolutne skore Generated:{total_score}")
print(f"Absolutne skore Recovered:{assumed_score}")

fig, ax = plt.subplots()
ax.plot(states[:100] + 2.5, label='recovered')
ax.plot(gen_states[:100], label='generated')
ax.set_yticks([])
ax.set_title('States compared to generated')
ax.set_xlabel('Time (# rolls)')
ax.set_xlabel('State')
ax.legend(loc="center left")
plt.savefig("tex/odhad_stavov.jpg")
plt.show()

