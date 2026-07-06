# ANALISIS DE DATOS, UTILIZANDO VISUALIZACIÓN DE DATOS Y CORRELACION PARA DETERMINAR QUÉ PERFIL DE PASAJERO DEL TITANIC
# TUVO LA MAYOR PROBABILIDAD DE SOBREVIVIR

# IMPORTAR LIBRERIAS
import os
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carpeta donde se guardan las figuras y el HTML (misma carpeta del script)
CARPETA = os.path.dirname(os.path.abspath(__file__))

# 1. CARGAR DATOS
df = pd.read_csv('titanic.csv')

# 2. ANALISIS EXPLORATORIO INICIAL (datos crudos)
print("="*60)
print("EXPLORACION INICIAL (datos crudos)")
print("="*60)
print(df.head())
print(df.info())
print(df.describe())
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 3. LIMPIEZA DE DATOS
df_limpio = df.copy()

# Age: nulos -> imputar con la mediana (menos sensible a outliers que la media)
df_limpio['Age'] = df_limpio['Age'].fillna(df_limpio['Age'].median())

# Embarked: nulos -> imputar con la moda (valor mas frecuente)
df_limpio['Embarked'] = df_limpio['Embarked'].fillna(
    df_limpio['Embarked'].mode()[0])

# Cabin: demasiados nulos (>75%) para imputar de forma confiable.
# En vez de descartar la columna por completo, se crea una variable binaria util:
# "el pasajero tenia cabina registrada o no" (puede correlacionar con la clase/tarifa).
df_limpio['Tiene_Cabina'] = df_limpio['Cabin'].notnull().astype(int)
df_limpio = df_limpio.drop(columns=['Cabin'])

# Duplicados
duplicados = df_limpio.duplicated().sum()
print(f"\nFilas duplicadas encontradas: {duplicados}")
df_limpio = df_limpio.drop_duplicates()

# Columnas que no aportan al analisis numerico (identificadores unicos)
df_limpio = df_limpio.drop(columns=['PassengerId', 'Name', 'Ticket'])

print("\nValores nulos despues de la limpieza:")
print(df_limpio.isnull().sum())

# 4. CODIFICACION DE VARIABLES CATEGORICAS
df_codificado = df_limpio.copy()
df_codificado['Sex'] = df_codificado['Sex'].map({'male': 0, 'female': 1})
df_codificado = pd.get_dummies(df_codificado, columns=[
                               'Embarked'], drop_first=True)

# Asegurar tipos numericos para correlacion (get_dummies puede crear bool)
for col in df_codificado.select_dtypes(include='bool').columns:
    df_codificado[col] = df_codificado[col].astype(int)

print("\nColumnas finales listas para analisis:")
print(df_codificado.columns.tolist())

# 5. ANALISIS EXPLORATORIO DE DATOS (EDA) - distribuciones generales

print("\nANALISIS EXPLORATORIO DE DATOS")

# Distribucion de la variable Survived
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Survived',
              hue='Survived', palette='Set2', legend=False)
plt.title('Distribución de Supervivencia')
plt.xlabel('Supervivencia')
plt.ylabel('Cantidad de Pasajeros')
plt.xticks([0, 1], ['No Sobrevivió', 'Sobrevivió'])
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "01_distribucion_supervivencia.png"), dpi=150)
plt.close()

# Distribucion por sexo
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Sex', hue='Sex',
              palette='Pastel1', legend=False)
plt.title('Distribución por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Cantidad de Pasajeros')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "02_distribucion_sexo.png"), dpi=150)
plt.close()

# Distribucion por clase
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Pclass', hue='Pclass',
              palette='Blues', legend=False)
plt.title('Distribución por Clase')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Pasajeros')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "03_distribucion_clase.png"), dpi=150)
plt.close()

# Distribucion de edades
plt.figure(figsize=(8, 5))
sns.histplot(df_limpio['Age'], bins=30, kde=True, color='skyblue')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "04_distribucion_edad.png"), dpi=150)
plt.close()

# Distribucion de tarifa pagada
plt.figure(figsize=(8, 5))
sns.histplot(df_limpio['Fare'], bins=30, kde=True, color='orange')
plt.title('Distribución de la Tarifa')
plt.xlabel('Tarifa')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "05_distribucion_tarifa.png"), dpi=150)
plt.close()

# Distribucion por puerto de embarque
plt.figure(figsize=(6, 4))
sns.countplot(data=df_limpio, x='Embarked', hue='Embarked',
              palette='Accent', legend=False)
plt.title('Distribución por Puerto de Embarque')
plt.xlabel('Puerto')
plt.ylabel('Cantidad de Pasajeros')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "06_distribucion_puerto.png"), dpi=150)
plt.close()

print("Graficas de distribucion guardadas en:", CARPETA)

# 6. ANALISIS DE CORRELACION
matriz_corr = df_codificado.corr(numeric_only=True)
print("\nCorrelacion de cada variable con 'Survived':")
print(matriz_corr['Survived'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlacion - Titanic")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "07_matriz_correlacion.png"), dpi=150)
plt.close()

# 7. VISUALIZACIONES - supervivencia cruzada con otras variables

# Supervivencia por sexo
plt.figure(figsize=(6, 4))
sns.barplot(data=df_limpio, x='Sex', y='Survived', errorbar=None)
plt.title("Tasa de supervivencia por sexo")
plt.ylabel("Proporcion de supervivencia")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "08_supervivencia_por_sexo.png"), dpi=150)
plt.close()

# Supervivencia por clase
plt.figure(figsize=(6, 4))
sns.barplot(data=df_limpio, x='Pclass', y='Survived', errorbar=None)
plt.title("Tasa de supervivencia por clase (Pclass)")
plt.ylabel("Proporcion de supervivencia")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "09_supervivencia_por_clase.png"), dpi=150)
plt.close()

# Supervivencia por puerto de embarque
plt.figure(figsize=(6, 4))
sns.barplot(data=df_limpio, x='Embarked', y='Survived', errorbar=None)
plt.title("Tasa de supervivencia por puerto de embarque")
plt.ylabel("Proporcion de supervivencia")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "10_supervivencia_por_puerto.png"), dpi=150)
plt.close()

# Distribucion de edad segun supervivencia
plt.figure(figsize=(7, 4))
sns.histplot(data=df_limpio, x='Age', hue='Survived',
             bins=30, kde=True, multiple='stack')
plt.title("Distribucion de edad segun supervivencia")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "11_edad_por_supervivencia.png"), dpi=150)
plt.close()

# Boxplot de tarifa (Fare) segun supervivencia
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_limpio, x='Survived',
            y='Fare', hue='Survived', legend=False)
plt.title("Tarifa pagada segun supervivencia")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA, "12_tarifa_por_supervivencia.png"), dpi=150)
plt.close()

print("Graficas de supervivencia guardadas en:", CARPETA)

# 8. TABLA HTML CON LOS DATOS YA LIMPIOS


def mostrar_tabla_datos_limpios(dataframe, abrir_navegador=True):
    """Muestra los datos limpios en una tabla HTML en el navegador."""
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

    output_path = os.path.join(CARPETA, "datos_limpios.html")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    if abrir_navegador:
        webbrowser.open(f"file://{output_path}")
    print(f"\nTabla abierta en el navegador: {output_path}")


if __name__ == "__main__":
    # abrir_navegador=False para que corra bien en servidores/notebooks sin display
    mostrar_tabla_datos_limpios(df_limpio, abrir_navegador=False)
    print("\nProceso completo. Revisa las imagenes .png y el archivo datos_limpios.html")
