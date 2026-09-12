import csv
from sklearn.metrics import cohen_kappa_score

A_fluency, M_fluency = [], []
A_relevance, M_relevance = [], []
A_answerability, M_answerability = [], []

with open("Human_evaluation.tsv", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        A_fluency.append(int(row["A_fluency"]))
        M_fluency.append(int(row["M_fluency"]))
        A_relevance.append(int(row["A_relevance"]))
        M_relevance.append(int(row["M_relevance"]))
        A_answerability.append(int(row["A_answerability"]))
        M_answerability.append(int(row["M_answerability"]))

print("Fluency % yes (A):", sum(A_fluency)/len(A_fluency)*100)
print("Fluency % yes (M):", sum(M_fluency)/len(M_fluency)*100)
print("Fluency kappa:", cohen_kappa_score(A_fluency, M_fluency))

print("Relevance % yes (A):", sum(A_relevance)/len(A_relevance)*100)
print("Relevance % yes (M):", sum(M_relevance)/len(M_relevance)*100)
print("Relevance kappa:", cohen_kappa_score(A_relevance, M_relevance))

print("Answerability % yes (A):", sum(A_answerability)/len(A_answerability)*100)
print("Answerability % yes (M):", sum(M_answerability)/len(M_answerability)*100)
print("Answerability kappa:", cohen_kappa_score(A_answerability, M_answerability))