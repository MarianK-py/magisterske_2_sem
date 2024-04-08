import pandas as pd
from pgmpy.models import NaiveBayes, BayesianNetwork
from pgmpy.inference import VariableElimination

# Marian Kravec

#Lod dataset
#exams = pd.read_csv('C:\Documents\Temp\cvikonaivebayes\exams.csv')
exams = pd.read_csv('exams.csv')

#model = NaiveBayes()
model = BayesianNetwork()
#add all independent evidence features
model.add_edges_from([('passed', 'has_time'), ('passed','wants_to'), ('passed', 'studying')])

#train the model to classify the 'passed' variable
#for other variables you might want to change the model
#model.fit(exams, 'passed')
model.fit(exams)

#predict most probable outcome based on evidence
inference = VariableElimination(model)

# Task 1
print("Task 1")
prediction_prob = inference.query(variables=['passed'], evidence = {'has_time':1,'wants_to':1,'studying':1}, show_progress=False).values
print("Probability of pass if has time, wants to learn and learn just enough:", prediction_prob[1])
print("Probability of pass if has time, wants to learn and learn just enough:", prediction_prob[0])
print()
# Task 2
print("Task 2")
prediction_prob = inference.query(variables=['passed'], evidence = {'has_time':0,'wants_to':1,'studying':0}, show_progress=False).values
print("Probability of pass if not have time, wants to learn and learn just a bit:", prediction_prob[1])
print("Probability of pass if not have time, wants to learn and learn just a bit:", prediction_prob[0])
print()
# Task 3
print("Task 3")
prediction_prob = inference.query(variables=['has_time'], evidence = {'passed':1,'wants_to':0,'studying':0}, show_progress=False).values
print("Probability of having time if passed, not wanted to learn and learn just a bit:", prediction_prob[1])
print("Probability of not having timeif passed, not wanted to learn and learn just a bit:", prediction_prob[0])
print()
# Task 4
print("Task 4")
prediction_prob = inference.query(variables=['wants_to'], evidence = {'passed':0,'has_time':1,'studying':2}, show_progress=False).values
print("Probability of wanted to learn if not passed, had time and learned a lot:", prediction_prob[1])
print("Probability of not wanted to learn if not passed, had time and learned a lot:", prediction_prob[0])
print()
# Task 5
print("Task 5")
prediction_prob = inference.query(variables=['studying'], evidence = {'passed':1,'has_time':1,'wants_to':0}, show_progress=False).values
print("Probability of learn just a bit if passed, not wanted to learn and had time:", prediction_prob[0])
print("Probability of learn just enough if passed, not wanted to learn and had time:", prediction_prob[1])
print("Probability of learn a lot if passed, not wanted to learn and had time:", prediction_prob[2])
print()

#look at the conditional probabilities. Think abou last week excercise.
#How you can caluclate the probabilities you want?
#for cpd in model.get_cpds():
#    print(cpd)
