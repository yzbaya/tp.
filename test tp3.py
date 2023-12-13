# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import graphviz 
from sklearn.tree import export_text

#  Exploration et préparation des données
data = pd.read_csv('jouer.csv')  # Charger le fichier CSV

#  Déclaration des variables X et Y
X = data.drop('jouer', axis=1)  
y = data['jouer'] 

#  Division du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Construction de l'arbre de décision avec Gini
dtree_gini = DecisionTreeClassifier(criterion='gini')
dtree_gini.fit(X_train, y_train)

#  Extraction des règles de décision
rules_gini = export_text(dtree_gini, feature_names=list(X.columns))
print(rules_gini)

#  Calcul de l'accuracy
y_pred_train = dtree_gini.predict(X_train)
y_pred_test = dtree_gini.predict(X_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy on train set: {accuracy_train}")
print(f"Accuracy on test set: {accuracy_test}")

# Vérification de l'Overfitting et Underfitting


# Étape 8 : Visualisation de l'arbre de décision avec Matplotlib
plt.figure(figsize=(12, 8))
export_graphviz(dtree_gini, out_file='tree_gini.dot', feature_names=X.columns, filled=True)
with open("tree_gini.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# Étape 8 : Visualisation de l'arbre de décision avec Graphviz
graph = graphviz.Source(dot_graph)
graph.render("tree_gini", format="png")

# Étape 9 : Construction de l'arbre de décision avec Entropy
dtree_entropy = DecisionTreeClassifier(criterion='entropy')
dtree_entropy.fit(X_train, y_train)

# Étape 10 : Affichage de la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(conf_matrix)

# Étape 11 : Affichage du rapport de classification
class_report = classification_report(y_test, y_pred_test)
print("Classification Report:")
print(class_report)
