import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans


sns.set(style="whitegrid")


df = pd.read_csv('si_env-2020.csv', sep=';', encoding='latin1')
df.columns = df.columns.str.strip() 


print(df.head())


df['data_hora_boletim'] = pd.to_datetime(df['data_hora_boletim'], errors='coerce')
df['hora'] = df['data_hora_boletim'].dt.hour
df['mes'] = df['data_hora_boletim'].dt.month
df['dia_semana'] = df['data_hora_boletim'].dt.day_name()


df.replace(['-', ' ', '', 'NA', 'nan', 'NaN'], np.nan, inplace=True)
df = df.dropna(subset=['desc_severidade', 'Idade', 'sexo'])  

cat_cols = ['sexo', 'cinto_seguranca', 'Embreagues', 'categoria_habilitacao',
            'descricao_habilitacao', 'especie_veiculo', 'desc_severidade']
for col in cat_cols:
    df[col] = df[col].astype(str)

plt.figure(figsize=(8, 5))
df['desc_severidade'].value_counts().plot(kind='bar')
plt.title("Acidentes por Severidade")
plt.xlabel("Tipo de Severidade")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.countplot(data=df, x='sexo', hue='desc_severidade')
plt.title("Acidentes por Sexo e Severidade")
plt.tight_layout()
plt.show()

bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '60+']
df['faixa_etaria'] = pd.cut(df['Idade'].astype(int), bins=bins, labels=labels)

sns.countplot(data=df, x='faixa_etaria', order=labels)
plt.title("Acidentes por Faixa Etária")
plt.tight_layout()
plt.show()

sns.countplot(data=df, x='cinto_seguranca', hue='desc_severidade')
plt.title("Uso de Cinto vs Severidade")
plt.tight_layout()
plt.show()

sns.countplot(data=df, x='Embreagues', hue='desc_severidade')
plt.title("Embriaguez e Severidade do Acidente")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='categoria_habilitacao', hue='desc_severidade')
plt.title("Categoria da Habilitação e Severidade")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='especie_veiculo', hue='desc_severidade')
plt.title("Tipo de Veículo Envolvido")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(data=df, x='dia_semana', order=order)
plt.title("Acidentes por Dia da Semana")
plt.tight_layout()
plt.show()



model_df = df[['Idade', 'hora', 'mes', 'sexo', 'cinto_seguranca', 'Embreagues',
               'categoria_habilitacao', 'especie_veiculo', 'desc_severidade']].dropna()


model_df = pd.get_dummies(model_df, drop_first=True)


X = model_df.drop('desc_severidade_MÉDIO', axis=1, errors='ignore')
y = df['desc_severidade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred))


conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

cluster_df = df[['Idade', 'hora']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

sns.scatterplot(data=df, x='Idade', y='hora', hue='cluster', palette='Set1')
plt.title("Clusters de Acidentes por Idade e Hora")
plt.tight_layout()
plt.show()
