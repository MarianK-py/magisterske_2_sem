from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, State
from pgmpy.sampling import BayesianModelSampling

## Autor: Marian Kravec

# b)
elektricky_model = BayesianNetwork(
    [
        ("Linka", "Ma 1 vozeň"),
        ("Čas", "Ma 1 vozeň"),
        ("Linka", "Ide na Zochovu"),
        ("Ide do depa", "Ide na Zochovu"),
        ("Linka", "Ide na Odbojarov"),
        ("Ide do depa", "Ide na Odbojarov"),
    ]
)

# Linka: 0: 9, 1: 4, 2: 5
# Čas: 0: deň, 1: večer
# ostatné: 0: FALSE, 1: TRUE

cpd_linka = TabularCPD(variable="Linka", variable_card=3, values=[[0.5], [0.3], [0.2]])
cpd_cas = TabularCPD(variable="Čas", variable_card=2, values=[[0.66], [0.34]])
cpd_depo = TabularCPD(variable="Ide do depa", variable_card=2, values=[[0.9], [0.1]])
cpd_1_vozen = TabularCPD(
    variable="Ma 1 vozeň",
    variable_card=2,
    values=[[1, 0.9, 0.8, 0.8, 0.1, 0],
            [0, 0.1, 0.2, 0.2, 0.9, 1]],
    evidence=["Linka", "Čas"],
    evidence_card=[3, 2],
)
cpd_zochova = TabularCPD(
    variable="Ide na Zochovu",
    variable_card=2,
    values=[[0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0]],
    evidence=["Linka", "Ide do depa"],
    evidence_card=[3, 2],
)
cpd_odbojarov = TabularCPD(
    variable="Ide na Odbojarov",
    variable_card=2,
    values=[[1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1]],
    evidence=["Linka", "Ide do depa"],
    evidence_card=[3, 2],
)

elektricky_model.add_cpds(cpd_linka, cpd_cas, cpd_depo, cpd_1_vozen, cpd_zochova, cpd_odbojarov)
inference = BayesianModelSampling(elektricky_model)

#c)
evidence_c = [State(var="Čas", state=1), #Večer
            State(var="Ma 1 vozeň", state=1) #Električka má jeden vozeň
            ]
sample_size_c = 10000
c = inference.rejection_sample(evidence=evidence_c, size=sample_size_c)
pocet_na_odborarov = sum(c["Ide na Odbojarov"])

pravd_na_odborarov = pocet_na_odborarov/sample_size_c

print(f"Pravdepodobnosť, že električka s jedným vozňom večer obslúži zastávku Odborárov je približne {100*pravd_na_odborarov:2.1f} %")

#d)
evidence_d = [State(var="Linka", state=0)] #Linka číslo 9
sample_size_d = 10000
c = inference.rejection_sample(evidence=evidence_d, size=sample_size_d)
pocet_s_jednym_voznom = sum(c["Ma 1 vozeň"])

pravd_jedneho_vozna = pocet_s_jednym_voznom/sample_size_d

print(f"Pravdepodobnosť, že električka 9 má iba jeden vozeň:      {100*pravd_jedneho_vozna:2.1f} %")
print(f"Pravdepodobnosť, že električka 9 má viac ako jeden vozeň: {100*(1-pravd_jedneho_vozna):2.1f} %")
print("S väčšou pravdepodobnosťou má viac ako jeden vozeň")