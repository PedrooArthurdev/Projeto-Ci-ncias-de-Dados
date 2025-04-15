import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('mobile_sales_data.csv')
df.dropna(subset=['Price', 'RAM', 'ROM', 'SSD'], inplace=True)
df.columns = [col.strip().replace(" ", "_") for col in df.columns]


def extract_number(x):
    if isinstance(x, str):
        x = x.upper().replace('GB', '').replace('TB', '000')  
        return pd.to_numeric(''.join(filter(str.isdigit, x)), errors='coerce')
    return x

df['RAM'] = df['RAM'].apply(extract_number)
df['ROM'] = df['ROM'].apply(extract_number)
df['SSD'] = df['SSD'].apply(extract_number)


df['Processor_Specification'] = df['Processor_Specification'].fillna('Unknown')
le = LabelEncoder()
df['Processor_Label'] = le.fit_transform(df['Processor_Specification'])


features = ['RAM', 'ROM', 'SSD', 'Processor_Label']
target = 'Price'

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


rf_pred = rf.predict(X_test)


print("🌳 Avaliação do Random Forest Regressor:")
print(f"R²: {r2_score(y_test, rf_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")


plt.scatter(y_test, rf_pred, color='green')
plt.xlabel("Preço Real")
plt.ylabel("Preço Predito")
plt.title("Random Forest - Preço Real vs Predito")
plt.grid(True)
plt.tight_layout()
plt.show()


importances = rf.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)


sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title("Importância das Variáveis no Modelo Random Forest")
plt.tight_layout()
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


sns.scatterplot(data=df, x='RAM', y='Price', hue='Cluster', palette='Set2')
plt.title('Clusterização de Produtos (RAM vs Price)')
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv('mobile_sales_data.csv')

df.drop_duplicates(inplace=True)
df.dropna(subset=['Price', 'Quantity Sold'], inplace=True)


df.columns = [col.strip().replace(" ", "_") for col in df.columns]

df['Revenue'] = df['Price'] * df['Quantity_Sold']

kpis = {
    'Receita total': df['Revenue'].sum(),
    'Unidades vendidas': df['Quantity_Sold'].sum(),
    'Ticket médio': df['Revenue'].sum() / df['Quantity_Sold'].sum(),
    'Produto mais vendido': df.loc[df['Quantity_Sold'].idxmax(), 'Product'],
    'Produto mais lucrativo': df.loc[df['Revenue'].idxmax(), 'Product'],
    'Produto mais caro': df.loc[df['Price'].idxmax(), 'Product'],
    'Produto mais barato': df.loc[df['Price'].idxmin(), 'Product'],
    'Marca mais vendida': df.groupby('Brand')['Quantity_Sold'].sum().idxmax(),
    'Marca com maior receita': df.groupby('Brand')['Revenue'].sum().idxmax(),
    'Localidade com mais vendas': df.groupby('Customer_Location')['Quantity_Sold'].sum().idxmax(),
}

print("\n🔹 KPIs:")
for k, v in kpis.items():
    print(f"{k}: {v}")



df.groupby('Brand')['Revenue'].sum().sort_values().plot(kind='barh', title='Receita por Marca')
plt.xlabel('Receita Total')
plt.tight_layout()
plt.show()


top10 = df.sort_values('Quantity_Sold', ascending=False).head(10)
sns.barplot(x='Quantity_Sold', y='Product', data=top10)
plt.title('Top 10 Produtos Mais Vendidos')
plt.tight_layout()
plt.show()

sns.scatterplot(x='Price', y='Quantity_Sold', hue='Brand', data=df)
plt.title('Preço vs Quantidade Vendida')
plt.tight_layout()
plt.show()

sns.boxplot(x='Brand', y='Price', data=df)
plt.title('Distribuição de Preços por Marca')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribuição de Preços')
plt.tight_layout()
plt.show()


sns.barplot(x='Region', y='Revenue', data=df, estimator=np.sum)
plt.title('Receita Total por Região')
plt.tight_layout()
plt.show()

top10_rev = df.sort_values('Revenue', ascending=False).head(10)
sns.barplot(x='Revenue', y='Product', data=top10_rev)
plt.title('Top 10 Produtos com Maior Receita')
plt.tight_layout()
plt.show()

corr = df[['Price', 'Quantity_Sold', 'Revenue']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlação entre variáveis')
plt.tight_layout()
plt.show()