# Analisis del Titanic - parte 2: clustering con K-Means
# La idea es agrupar a los pasajeros en perfiles parecidos SIN usar la
# columna Survived, y ya con los grupos armados ver cuanto sobrevivio
# cada uno. Asi salen los perfiles con mas y menos supervivencia.
# Las graficas se guardan en la carpeta graficas

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Seccion 1: rutas y carpeta de salida

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_GRAFICAS = os.path.join(CARPETA, "graficas")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)


def guardar_grafica(nombre):
    # guarda la figura actual y la cierra
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, nombre + ".png"), dpi=150)
    plt.close()


# Seccion 2: cargar y limpiar
# misma limpieza que en main.py para que los dos analisis usen los
# mismos datos. Este script corre solo, no necesita que main corra antes

df = pd.read_csv(os.path.join(CARPETA, "titanic.csv"))

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Tiene_Cabina'] = df['Cabin'].notnull().astype(int)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])
df = df.drop_duplicates()

# Seccion 3: preparar los datos para el clustering
# estas variables son las que describen al pasajero. Ojo: Survived no
# entra aqui, los grupos se tienen que formar solos y la supervivencia
# la comparamos hasta el final

variables_perfil = ['Pclass', 'Sex', 'Age', 'SibSp',
                    'Parch', 'Fare', 'Tiene_Cabina']
X = df[variables_perfil]

# hay que estandarizar porque kmeans se basa en distancias, si no
# Fare (llega hasta 500) aplastaria a Sex que solo vale 0 o 1
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# Seccion 4: metodo del codo para elegir k
# pruebo de 1 a 10 clusters y grafico la inercia, donde la curva
# deja de bajar fuerte ahi esta el buen numero de grupos

inercias = []
rango_k = range(1, 11)
for k in rango_k:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X_escalado)
    inercias.append(modelo.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(rango_k, inercias, marker='o')
plt.title('Método del codo para elegir k')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.xticks(rango_k)
plt.grid(alpha=0.3)
guardar_grafica("13 metodo del codo")

# Seccion 5: entrenar kmeans
# en la grafica del codo se ve que despues de k=4 ya casi no mejora,
# por eso van 4 grupos. El random_state fijo es para que salga lo
# mismo cada vez que se corre

K_ELEGIDO = 4
kmeans = KMeans(n_clusters=K_ELEGIDO, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_escalado)

# Seccion 6: perfil de cada cluster
# promedio de cada variable por grupo, con esto se describe quien es
# quien, tipo hombres de 3ra clase con tarifa baja

print("=" * 60)
print("PERFIL PROMEDIO DE CADA CLUSTER")
print("=" * 60)
perfil = df.groupby('Cluster')[variables_perfil + ['Survived']].mean().round(2)
perfil['Pasajeros'] = df.groupby('Cluster').size()
print(perfil)

# Seccion 7: tasa de supervivencia por cluster
# este es el resultado importante del proyecto, comparar que grupo
# sobrevivio mas

tasa = df.groupby('Cluster')['Survived'].mean()
print("\nTasa de supervivencia por cluster:")
print((tasa.sort_values(ascending=False) * 100).round(1).astype(str) + " %")

# ojo: las barras van en orden de cluster (0,1,2,3) y las etiquetas
# se colocan en ese mismo orden para que cada % caiga sobre su barra
plt.figure(figsize=(7, 4))
sns.barplot(x=tasa.index, y=tasa.values, hue=tasa.index,
            palette='viridis', legend=False)
plt.title('Tasa de supervivencia por cluster (perfil de pasajero)')
plt.xlabel('Cluster')
plt.ylabel('Proporción que sobrevivió')
plt.ylim(0, 1)
for i, v in enumerate(tasa.values):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
guardar_grafica("14 supervivencia por cluster")

# Seccion 8: ver los clusters en 2d
# dispersion de edad contra tarifa coloreada por cluster, sirve para
# mostrar los grupos en el documento

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Cluster',
                palette='viridis', alpha=0.7)
plt.title('Clusters de pasajeros (Edad vs Tarifa)')
plt.xlabel('Edad')
plt.ylabel('Tarifa pagada')
guardar_grafica("15 clusters edad vs tarifa")

print("\nListo, graficas 13, 14 y 15 guardadas en:", CARPETA_GRAFICAS)
