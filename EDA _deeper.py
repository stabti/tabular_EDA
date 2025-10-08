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
- Étudiez ce script et comprenez-le, jouez avec les paramètres du clustering
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

# Reset index to have 'timestamp' as a column
X_reset = X.reset_index()


# Select only numeric columns for scatter matrix (excluding timestamp)
numeric_cols = X_reset.select_dtypes(include=np.number).columns

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
- Que resprésente ce dataset ? Faites des recherches sur internet pour mieux comprendre son contexte.
- Quelles cas d'usages on peut imaginer avec ce dataset ?
- De quand à quand ont été prises les mesures ?
- Est ce que les données sont bien temporellemnt ordonnées ?
- Est ce qu'il y a des doublons ?
- Les valeurs manquantes sont égales à -200 sur ce dataset, je les ai remplacées avec une méthode basique. Vérifiez qu'elle est pertinente. Si non, essayez d'autres méthodes.
- Est ce qu'il y a des valeurs aberrantes ? (utiliser des visualisations adaptées)
- Comment évoluent les données en une journée ? une semaine ? une saison ? (utiliser des visualisations adaptées, commencez par visualiser les données brutes)
- Quelles sont les variables les plus corrélées entre elles ?
"""


def plot_raw_between(df, start_date, end_date, cols=None, figsize=(12,6), ylabel='Valeur', title=None):
    """
    Affiche les séries temporelles (lignes pleines) pour les colonnes demandées
    entre start_date et end_date (inclus).
    - df : DataFrame avec index DatetimeIndex
    - start_date, end_date : str ou datetime (ex: '2004-03-10', '2004-03-10 12:00')
    - cols : liste de colonnes à afficher (None => toutes les colonnes numériques)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Le DataFrame doit avoir un index de type DatetimeIndex.")
    start = pd.to_datetime(start_date, errors='coerce')
    end = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start_date ou end_date invalide / non parsable.")
    if start > end:
        start, end = end, start  # swap si ordre inversé (tolérance)
    sub = df.loc[start:end]
    if sub.empty:
        print(f"Aucune donnée entre {start} et {end}.")
        return
    if cols is None:
        cols = sub.select_dtypes(include=np.number).columns.tolist()
    else:
        # garder seulement les colonnes existantes
        cols = [c for c in cols if c in sub.columns]
        if not cols:
            print("Aucune des colonnes demandées n'est présente dans le DataFrame.")
            return

    plt.figure(figsize=figsize)
    for col in cols:
        plt.plot(sub.index, sub[col], '-', linewidth=1, label=col)
    plt.xlabel('Timestamp')
    plt.ylabel(ylabel)
    if title is None:
        title = f"Données brutes entre {start.date()} et {end.date()}"
    plt.title(title)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Exemple d'appel d'affichage données brutes entre deux dates
plot_raw_between(X, '2004-03-10', '2004-03-20', cols=['CO(GT)', 'T'], figsize=(14,6))
