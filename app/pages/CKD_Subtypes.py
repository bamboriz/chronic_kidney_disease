import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data_utils import load_data, preprocess_continuous_data, impute_missing_categorical_data, preprocess, get_inertia

st.title('Exploring Potential Types of CKD')

df = load_data()
cols_names = list(df.columns)

col1, col2 = st.columns(2)
with col1:
   target = st.selectbox(
    'Please select target variable:',
    cols_names,
    len(cols_names)-1)

with col2:
    filter_target = st.selectbox(
    'Filter',
    df[target].unique())

df = load_data(target, filter_target)
st.write(df.sample(5))
st.write(f"We have {df.shape[1]} columns and {df.shape[0]} _{filter_target}_ samples with Chronic Kidney Disease in our dataset.")

st.write('\n')

st.subheader('Data Preprocessing')
st.write('We process the data in 3 main steps: Handling missing values, Encoding categorical features and then Scaling.')
st.markdown(
    '''
    - We impute categorical missing values with the **mode** of each column and numerical missing values with the **mean**.
    - For scaling, we will use the **standard scaler** to normalise our features.
    '''
)
df = preprocess_continuous_data(df)
df = impute_missing_categorical_data(df)
st.write(df.head())
st.markdown('<spand style="text-align: center">_NB: Not displaying the encoding step as it results in sparse data that is problematic to display in streamlit._</span>', unsafe_allow_html=True)

st.write('\n')

st.subheader('Clustering')

df = load_data(target, filter_target)
X = preprocess(df)
num_clusters, distortions = get_inertia(X, 11)
fig = px.line(
            x=num_clusters, y=distortions, markers=True, 
            title = 'Figure: Elbow chart to determine optimal number of clusters.',
             width=600, height=400,
             labels={ "x": "Number of Clusters",  "y": "Inertia"}
            )
st.write(fig)
st.markdown('''
                From chart, the optimal value of clusters is not very obvious. Let us use **Principal Component Analysis** to 
                see if we can denoise the data to obtain a more obvious value. Try different number of components to see how they affect the graph.
            ''')

number_of_features = len(df.columns)
number_of_components = st.slider('Please select the number of principal components to use:', 1, number_of_features, 2)
pca = PCA(n_components=number_of_components)
pca.fit(X)
variability_explained = np.cumsum(pca.explained_variance_ratio_ * 100)[-1]
st.markdown(f'**{number_of_components}** components explain **{round(variability_explained, 2)}%** of variability in the data')

X_pca=pca.fit_transform(X)
num_clusters, distortions = get_inertia(X_pca, 11)
fig = px.line(
            x=num_clusters, y=distortions, markers=True, 
            title = 'Figure: Elbow chart to determine optimal number of clusters.',
             width=600, height=400,
             labels={ "x": "Number of Clusters",  "y": "Inertia"}
            )
st.write(fig)
st.markdown('After applying a PCA, **3 clusters** looks to be the optimal value.')

number_of_clusters = st.slider('Please select the number of cluster to use:', 2, number_of_features, 3)
km = KMeans(
    n_clusters=number_of_clusters, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_pred = km.fit_predict(X_pca)

st.write('\n')

st.subheader('Cluster Visualisation')

if number_of_components > 1:
    fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1], color=y_pred,
                title = 'Figure: Visualisation of clusters using 2 PCA components.',
                labels={ "x": "PCA 1",  "y": "PCA 2", "color": "Cluster"}
                )
    st.write(fig)
else:
    st.markdown("*Select at least 2 PCA components to have a visualisation of the clusters ...*")
    st.write('\n')

st.caption('Quick Statistics')
cluster_list = ['Cluster '+ str(i) for i in range(number_of_clusters)]
cluster_stats = st.selectbox(
    'Please a cluster:',
    cluster_list)

df_pred = df.copy()
df_pred['cluster'] = y_pred
index = int(cluster_stats.split(' ')[-1])
st.write(df_pred[df_pred['cluster'] == index].describe())


