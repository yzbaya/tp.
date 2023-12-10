import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Chargement des données
fruits = pd.read_csv('fruits.csv')  # Assurez-vous de spécifier le bon chemin vers votre fichier

label_names = {1: ' Pomme', 2: ' Mandarine', 3: ' Orange', 4: ' Citron'}

# Renommer la colonne de l'étiquette pour une meilleure compréhension
fruits['fruit_name'] = fruits['fruit_label'].map(label_names)

# Déclarer les variables caractéristiques X et la variable cible Y
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# Diviser le dataset en données d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Créer un modèle classificateur KNN avec K=3
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Calculer le score d'échantillons bien classifiés sur le jeu de données de test
accuracy = knn_model.score(X_test, y_test)
print(f"Score d'échantillons bien classifiés sur le jeu de données de test : {accuracy:.2f}")

# Prédire le nom d'un fruit pour de nouvelles valeurs
fruit_1 = knn_model.predict([[20, 4.4, 5.5]])  # Fruit 1
fruit_2 = knn_model.predict([[180, 8.0, 6.8]])  # Fruit 2

print(f"Prédiction pour Fruit 1 :{label_names[fruit_1[0]]}")
print(f"Prédiction pour Fruit 2 :{label_names[fruit_2[0]]}")

# Calculer la valeur de la métrique d'évaluation Accuracy et F1 score
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy : {accuracy:.2f}")
print(f"F1 Score : {f1:.2f}")
