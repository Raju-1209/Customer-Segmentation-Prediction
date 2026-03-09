# import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# page setup
st.set_page_config(page_icon=":material/public:", page_title="Customer Segmentation", layout="wide")

# load the data
st.subheader("Upload your dataset")
st.write("Please upload a CSV file containing your customer data.")
file = st.file_uploader("Choose a CSV file", type="csv")
df = None
if file:
    df = pd.read_csv(file)

with st.sidebar:
    st.title("Customer Segmentation")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    if df is not None:
        features = st.multiselect("Select features for clustering", options=df.columns, default=["Annual Income (k$)", "Spending Score (1-100)"])
        df = df.loc[:, features]

def preprocessing(df):
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = encoder.fit_transform(df[col])


def elbow():
    out = []
    k_values = range(1, 11)

    for i in k_values:
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(df)
        out.append(model.inertia_)

    KL = KneeLocator(k_values, out, curve="convex", direction="decreasing")
    df1 = pd.DataFrame({"K_val": k_values, "inertia": out})

    st.subheader("Elbow Method")
    fig = st.line_chart(data=df1, x="K_val", y="inertia")

    return KL.elbow

if df is not None:
    st.subheader('Samples of the data uploaded for visualization and clustering')
    st.write(df.sample(10))

    preprocessing(df)

    #optimized K value
    K = elbow()

    #model training
    model = KMeans(n_clusters=K, random_state=42)
    model.fit(df)
    labels = model.labels_
    df['cluster'] = labels

    #visualization
    st.subheader("Clustered Data")
    st.scatter_chart(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", color="cluster")