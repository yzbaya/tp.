import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets

# Chargement des données
iris_data = pd.read_csv('Iris.csv')

# Sélection des deux premiers attributs
X = iris_data[['SepalLengthCm', 'SepalWidthCm']]
y = iris_data['Species']

# Division du dataset en données d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Entrainement du modèle SVM linéaire
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Score sur le jeu de données de test
accuracy = svm_model.score(X_test, y_test)
print(f"Score d'échantillons bien classifiés sur le jeu de données de test : {accuracy:.2f}")

# Visualisation de la surface de décision
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k')
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.title('Surface de décision SVM linéaire')
plt.show()