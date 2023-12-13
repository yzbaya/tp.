import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# Charger le jeu de données
data = pd.read_csv('loan.csv')

# Explorer les données en utilisant un countplot pour la variable 'purpose'
sns.countplot(x='purpose', data=data)
# Convertir la colonne 'purpose' en variables dummies
data = pd.get_dummies(data, columns=['purpose'])
# Définir les caractéristiques (features) X et la variable cible (target) Y
X = data.drop('purpose', axis=1)  
Y = data['installement'] 
# Diviser le dataset en données d'apprentissage et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Créer le modèle Naive Bayes Gaussian
model = GaussianNB()

# Entraîner le modèle
model.fit(X_train, Y_train)
# Prédire les valeurs sur les données de test
Y_pred = model.predict(X_test)

# Calculer l'Accuracy et le score F1
accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

# Interprétation des résultats
print(f"Accuracy : {accuracy}")
print(f"F1 Score : {f1}")
