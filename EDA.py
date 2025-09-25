from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

""" Instructions :
- Utiliser "Run in interactive window " pour faire tourner ce script pas à pas
- Le script actuel permet de démarrer les analyses de manière "brute"
- Répondez aux questions en commentaire à la fin du script pour aller plus loin dans l'analyse"""

# fetch dataset 
air_quality = fetch_ucirepo(id=360) 
  
# data (as pandas dataframes) 
X = air_quality.data.features 
y = air_quality.data.targets 
  
# metadata 
print(air_quality.metadata) 
  
# variable information 
print(air_quality.variables) 



# Display first few rows of the features dataframe
print("First elements of the dataframe:\n", X.head())

# Merge the first two columns into a single timestamp column
X['timestamp'] = pd.to_datetime(X.iloc[:, 0] + ' ' + X.iloc[:, 1], errors='coerce')

# Optionally, drop the original Date and Time columns
X = X.drop(X.columns[[0, 1]], axis=1)

# Set timestamp as index (optional, for time series analysis)
X = X.set_index('timestamp')

print("First elements of th dataframe with better formatted timestamps:\n", X.head())
print("Last elements of th dataframe with better formatted timestamps:\n", X.tail())

# Est-ce que les timestamps sont bien ordonnés chronologiquement ?
is_sorted = X.index.is_monotonic_increasing
print("Les timestamps sont-ils ordonnés chronologiquement ? :", is_sorted)


# descriptive statistics
print(X.describe())

# Size of the dataset
print("Size of the dataset: ", X.shape)

# Reset index to have 'timestamp' as a column
X_reset = X.reset_index()

# Select only numeric columns for scatter matrix (excluding timestamp)
numeric_cols = X_reset.select_dtypes(include=np.number).columns

# 1. Afficher un pairplot entre les variables avec seaborn
sns.pairplot(X)
plt.suptitle("Pairplot des variables", y=1.02)
plt.show()


# 2. Compter les valeurs manquantes
# Détection et comptage des valeurs manquantes codées -200
missing_by_col = (X == -200).sum()
print("Nombre de valeurs -200 (valeurs manquantes) par colonne :\n", missing_by_col)

# Pourcentage de valeurs -200 par colonne
missing_pct_by_col = 100 * (X == -200).sum() / len(X)
print("Pourcentage de valeurs -200 par colonne :\n", missing_pct_by_col)

# Suppression de la colonne comportant 90% de valeurs manquantes
X = X.drop(columns=['NMHC(GT)'])

# Remplacer les -200 par NaN pour toutes les colonnes
X = X.replace(-200, np.nan)

# Ensuite, l'imputation fonctionnera correctement
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X[numeric_cols])

# 3. Afficher une matrice de corrélations (voir si j'impute ou pas avant)
corr_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice de corrélations")
plt.show()



# 2. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 3. Réduction de dimension (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Clustering (K-Means)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 5. Visualisation des clusters
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clustering K-Means après PCA')
plt.colorbar(label='Cluster')
plt.show()


# 
# 3. Réduction de dimension (UMAP)
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 4. Clustering (K-Means)
kmeans_umap = KMeans(n_clusters=5, random_state=42)
clusters = kmeans_umap.fit_predict(X_umap)

# 5. Visualisation des clusters
plt.figure(figsize=(8,6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Clustering K-Means après UMAP')
plt.colorbar(label='Cluster')
plt.show()


""" 
Questions à se poser sur le dataset :
- Que resprésente ce dataset ?
- Quelles cas d'usages on peut imaginer avec ce dataset ?
- De quand à quand ont été prises les mesures ?
- Est ce que les données sont bien temporellemnt ordonnées ?
- Est ce qu'il y a des doublons ?
- Les valeurs manquantes sont égales à -200 sur ce dataset, identifiez les et remplacez les avec une méthode d'imputation de votre choix.
- Est ce qu'il y a des valeurs aberrantes ? (utiliser des visualisations adaptées)
- Comment évoluent les données en une journée ? une semaine ? une saison ? (utiliser des visualisations adaptées)
- Quelles sont les variables les plus corrélées entre elles ?
"""