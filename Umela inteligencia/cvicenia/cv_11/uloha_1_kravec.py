from hmmlearn import hmm
import numpy as np

# Marian Kravec

gen_model = hmm.CategoricalHMM(n_components=4, random_state=42)

# kazda kombinacia pociatocnych prepnuti prepinacov je rovnaka
gen_model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])

# prepinanie pacok opicou, nic sa nezmeni na 49%, jedna packa sa zmeni na 21% (pre kazdu packu) a obe na 9%
gen_model.transmat_ = np.array([[0.49, 0.21, 0.21, 0.09],  # obycajna
                                [0.21, 0.49, 0.09, 0.21],# orieskova
                                [0.21, 0.09, 0.49, 0.21],# karamelova
                                [0.09, 0.21, 0.21, 0.49]]) # ories karamelova

# emisie podla zadania
gen_model.emissionprob_ = np.array([[0.8, 0.2],  # obycajna
                                    [0.3, 0.7],  # orieskova
                                    [0.2, 0.8],  # karamelova
                                    [0.6, 0.4]]) # ories karamelova

o_t = np.diag(gen_model.emissionprob_[:,0])
o_s = np.diag(gen_model.emissionprob_[:,1])

f_00 = gen_model.startprob_
f_01 = f_00 @ gen_model.transmat_ @ o_t
f_02 = f_01 @ gen_model.transmat_ @ o_s

b_22 = np.ones(4)
b_12 = gen_model.transmat_ @ o_s @ b_22
b_02 = gen_model.transmat_ @ o_t @ b_12

T_1 = (f_01*b_12)/np.sum(f_01*b_12)
T_2 = (f_02*b_22)/np.sum(f_02*b_22)

model_vysl = gen_model.predict_proba(np.array([0,1]).reshape(-1, 1))

print("Stav T_1:")
print("Nas vypocet:")
print(T_1)
print("Vypocet modelu:")
print(model_vysl[0,:])
print()
print("Stav T_2:")
print("Nas vypocet:")
print(T_2)
print("Vypocet modelu:")
print(model_vysl[1,:])
