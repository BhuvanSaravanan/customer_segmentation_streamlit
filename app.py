import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Dynamic Customer Segmentation & Business Insights")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\mailb\OneDrive\Desktop\Sigma Induction\Customer_Segmentation_Dataset.csv")
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True, errors='coerce')
    return df

df = load_data()

# --- Feature Engineering ---
df['Age'] = 2025 - df['Year_Birth']
df['Family_Size'] = df['Kidhome'] + df['Teenhome']
df['Customer_Tenure'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days // 365

product_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
df['Total_Spent'] = df[product_cols].sum(axis=1)
df['Fav_Product'] = df[product_cols].idxmax(axis=1)

df['Total_Accepted_Campaigns'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
df['Promotion_Responsive'] = df['Total_Accepted_Campaigns'] > 0

df['Total_Purchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']

# --- Clustering Controls ---
st.sidebar.header("Clustering Controls")
k = st.sidebar.slider("Number of clusters (K)", 2, 8, 4)

# --- KMeans Clustering on main features ---
cluster_features = ['Age','Income','Total_Spent','Recency','Family_Size','Customer_Tenure']
cluster_df = df[cluster_features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df.loc[cluster_df.index, 'Cluster'] = clusters.astype(int)

# --- People Section ---
st.subheader("People")
fig, ax = plt.subplots(figsize=(7, 3))
sns.histplot(data=df, x='Age', hue='Cluster', bins=20, palette='tab10', multiple='stack', ax=ax)
ax.set_title("Customer Age Distribution by Cluster", fontsize=14)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Number Of Customers", fontsize=12)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
# Move legend outside the plot and make sure it's always visible
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(
        handles=handles, labels=labels, title='Cluster',
        fontsize=10, title_fontsize=11,
        bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.
    )
st.pyplot(fig)
# --- Products Section ---
st.subheader("Products")
fig, ax = plt.subplots(figsize=(7, 3))
sns.countplot(data=df, x='Fav_Product', hue='Cluster', palette='tab10', ax=ax)
ax.set_title("Favorite Product Category by Cluster", fontsize=14)
ax.set_xlabel("Favorite Product", fontsize=12)
ax.set_ylabel("Number Of Customers", fontsize=12)
ax.tick_params(axis='x', labelsize=10, rotation=30)
ax.tick_params(axis='y', labelsize=10)
ax.legend(title='Cluster', fontsize=10, title_fontsize=11)
st.pyplot(fig)

# --- Promotion Section ---
st.subheader("Promotion")
promo_means = df.groupby('Cluster')['Total_Accepted_Campaigns'].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 3))
sns.lineplot(data=promo_means, x='Cluster', y='Total_Accepted_Campaigns', marker='o', color='tab:blue', ax=ax)
ax.set_title("Average Accepted Campaigns by Cluster", fontsize=14)
ax.set_xlabel("Cluster", fontsize=12)
ax.set_ylabel("Avg Accepted Campaigns", fontsize=12)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
st.pyplot(fig)

# --- Place Section ---
st.subheader("Place")
place_means = df.groupby('Cluster')[['NumWebPurchases','NumCatalogPurchases','NumStorePurchases']].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 3))
place_means_melted = place_means.melt(id_vars='Cluster', var_name='Channel', value_name='Avg Purchases')
sns.barplot(data=place_means_melted, x='Channel', y='Avg Purchases', hue='Cluster', palette='tab10', ax=ax)
ax.set_title("Average Purchases by Channel and Cluster", fontsize=14)
ax.set_xlabel("Channel", fontsize=12)
ax.set_ylabel("Average Purchases", fontsize=12)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.legend(title='Cluster', fontsize=10, title_fontsize=11)
st.pyplot(fig)