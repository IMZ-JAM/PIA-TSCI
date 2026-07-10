# Analisis del Titanic - parte 1
# Aqui va la limpieza, la exploracion y la correlacion.
# que perfil de pasajero sobrevivio mas?
# Todas las graficas se guardan en la carpeta graficas lo  modifique porque generaba muchos archivos

import os
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seccion 1: rutas y carpeta de salida
# uso la carpeta donde esta el script para que no truene si lo corres
# desde otro lado, y si no existe la carpeta de graficas se crea sola

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_GRAFICAS = os.path.join(CARPETA, "graficas")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)


def guardar_grafica(nombre):
    # guarda la figura actual y la cierra, para no repetir esto 12 veces
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICAS, nombre + ".png"), dpi=150)
    plt.close()


# Seccion 2: cargar los datos
# son 891 pasajeros del Titanic con 12 columnas

df = pd.read_csv(os.path.join(CARPETA, "titanic.csv"))

# Seccion 3: exploracion inicial
# antes de limpiar hay que ver que trae el dataset, cuantos nulos hay, etc

print("=" * 60)
print("EXPLORACION INICIAL (datos crudos)")
print("=" * 60)
print(df.head())
print(df.info())
print(df.describe())
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Seccion 4: limpieza

df_limpio = df.copy()

# Age trae nulos, le pongo la mediana porque aguanta mejor los
# valores extremos que el promedio
df_limpio['Age'] = df_limpio['Age'].fillna(df_limpio['Age'].median())

# Embarked solo trae 2 nulos, se rellenan con el valor mas comun
df_limpio['Embarked'] = df_limpio['Embarked'].fillna(
    df_limpio['Embarked'].mode()[0])

# Cabin viene vacia en mas del 75% de las filas, imposible rellenar eso.
# En lugar de tirar la columna la convierto en: tenia cabina registrada o no
df_limpio['Tiene_Cabina'] = df_limpio['Cabin'].notnull().astype(int)
df_limpio = df_limpio.drop(columns=['Cabin'])

# checar duplicados y quitarlos si hay
duplicados = df_limpio.duplicated().sum()
print(f"\nFilas duplicadas encontradas: {duplicados}")
df_limpio = df_limpio.drop_duplicates()

# estas columnas son puros identificadores, no dicen nada del pasajero
df_limpio = df_limpio.drop(columns=['PassengerId', 'Name', 'Ticket'])

print("\nValores nulos despues de la limpieza:")
print(df_limpio.isnull().sum())

# Seccion 5: pasar texto a numeros
# la correlacion solo jala con numeros, entonces Sex queda 0/1
# y Embarked se convierte en columnas dummy

df_codificado = df_limpio.copy()
df_codificado['Sex'] = df_codificado['Sex'].map({'male': 0, 'female': 1})
df_codificado = pd.get_dummies(df_codificado, columns=['Embarked'],
                               drop_first=True)

# get_dummies a veces deja columnas booleanas, las paso a enteros
for col in df_codificado.select_dtypes(include='bool').columns:
    df_codificado[col] = df_codificado[col].astype(int)

print("\nColumnas finales listas para analisis:")
print(df_codificado.columns.tolist())

# Seccion 6: graficas de distribucion
# 6 graficas para conocer los datos antes de analizarlos en serio

print("\nGenerando graficas de distribucion...")

# cuantos sobrevivieron y cuantos no
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Survived',
              hue='Survived', palette='Set2', legend=False)
plt.title('Distribución de Supervivencia')
plt.xlabel('Supervivencia')
plt.ylabel('Cantidad de Pasajeros')
plt.xticks([0, 1], ['No Sobrevivió', 'Sobrevivió'])
guardar_grafica("01 distribucion de supervivencia")

# cuantos hombres y mujeres viajaban
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Sex', hue='Sex',
              palette='Pastel1', legend=False)
plt.title('Distribución por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Cantidad de Pasajeros')
guardar_grafica("02 distribucion por sexo")

# pasajeros por clase
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Pclass', hue='Pclass',
              palette='Blues', legend=False)
plt.title('Distribución por Clase')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Pasajeros')
guardar_grafica("03 distribucion por clase")

# histograma de edades
plt.figure(figsize=(8, 5))
sns.histplot(df_limpio['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
guardar_grafica("04 distribucion de edad")

# histograma de tarifas
plt.figure(figsize=(8, 5))
sns.histplot(df_limpio['Fare'], bins=30, kde=True, color='orange')
plt.title('Distribución de la Tarifa')
plt.xlabel('Tarifa')
plt.ylabel('Frecuencia')
guardar_grafica("05 distribucion de tarifa")

# cuantos subieron en cada puerto
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Embarked', hue='Embarked',
              palette='Accent', legend=False)
plt.title('Distribución por Puerto de Embarque')
plt.xlabel('Puerto')
plt.ylabel('Cantidad de Pasajeros')
guardar_grafica("06 distribucion por puerto")

# Seccion 7: correlacion
# aqui vemos que variables se mueven junto con Survived, que es
# lo que nos interesa para el proyecto

matriz_corr = df_codificado.corr(numeric_only=True)
print("\nCorrelacion de cada variable con Survived:")
print(matriz_corr['Survived'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación - Titanic")
guardar_grafica("07 matriz de correlacion")

# Seccion 8: supervivencia cruzada con otras variables
# comparar directamente quien sobrevivio segun sexo, clase, etc

print("Generando graficas de supervivencia...")

# proporcion que sobrevivio por sexo
plt.figure(figsize=(6, 4))
sns.barplot(data=df_limpio, x='Sex', y='Survived', errorbar=None)
plt.title("Tasa de supervivencia por sexo")
plt.ylabel("Proporción de supervivencia")
guardar_grafica("08 supervivencia por sexo")

# proporcion que sobrevivio por clase
plt.figure(figsize=(6, 4))
sns.barplot(data=df_limpio, x='Pclass', y='Survived', errorbar=None)
plt.title("Tasa de supervivencia por clase")
plt.ylabel("Proporción de supervivencia")
guardar_grafica("09 supervivencia por clase")

# proporcion que sobrevivio segun el puerto donde subio
plt.figure(figsize=(6, 4))
sns.barplot(data=df_limpio, x='Embarked', y='Survived', errorbar=None)
plt.title("Tasa de supervivencia por puerto de embarque")
plt.ylabel("Proporción de supervivencia")
guardar_grafica("10 supervivencia por puerto")

# edades de los que vivieron vs los que no
plt.figure(figsize=(7, 4))
sns.histplot(data=df_limpio, x='Age', hue='Survived',
             bins=30, kde=True, multiple='stack')
plt.title("Distribución de edad según supervivencia")
guardar_grafica("11 edad segun supervivencia")

# tarifas de los que vivieron vs los que no
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_limpio, x='Survived', y='Fare',
            hue='Survived', legend=False)
plt.title("Tarifa pagada según supervivencia")
guardar_grafica("12 tarifa segun supervivencia")

print("Graficas guardadas en:", CARPETA_GRAFICAS)

# Seccion 9: tabla html con los datos limpios
# genera una pagina sencilla para revisar la base ya limpia en el navegador


def mostrar_tabla_datos_limpios(dataframe, abrir_navegador=True):
    html_table = dataframe.to_html(index=False, border=0, classes="data-table")
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Titanic - Datos limpios</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
      margin: 24px;
      background: #f5f5f5;
      color: #222;
    }}
    h1 {{ color: #2c3e50; margin-bottom: 8px; }}
    .info {{ margin-bottom: 16px; color: #555; }}
    .table-wrap {{
      overflow: auto;
      max-height: 85vh;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: white;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{
      background: #2c3e50;
      color: white;
      padding: 10px 8px;
      text-align: left;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    td {{ padding: 8px; border-bottom: 1px solid #eee; white-space: nowrap; }}
    tr:nth-child(even) {{ background: #f0f4f8; }}
    tr:hover {{ background: #e8f4fc; }}
  </style>
</head>
<body>
  <h1>Titanic — Datos limpios (0 nulos)</h1>
  <p class="info">
    Registros: <strong>{len(dataframe)}</strong> (originalmente {len(df)} pasajeros, {duplicados} duplicados eliminados)
  </p>
  <div class="table-wrap">{html_table}</div>
</body>
</html>"""

    output_path = os.path.join(CARPETA, "datos limpios.html")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    if abrir_navegador:
        webbrowser.open(f"file://{output_path}")
    print(f"\nTabla generada: {output_path}")


if __name__ == "__main__":
    mostrar_tabla_datos_limpios(df_limpio, abrir_navegador=False)
    print("\nListo, revisa la carpeta graficas y el archivo datos limpios.html")
