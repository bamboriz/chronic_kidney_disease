import streamlit as st
import plotly.express as px

from data_utils import load_data, get_numeric_cols, get_categorical_cols

st.title('Exploring Potential Risk Factors for CKD')

df = load_data()
cols_names = list(df.columns)
categorical_cols = get_categorical_cols()
numeric_cols = get_numeric_cols()
print(cols_names)
print(numeric_cols)
print(categorical_cols)

target = st.selectbox(
    'Please select target variable:',
    cols_names,
    len(cols_names)-1)
st.write(df.head())
st.write(f"We have {df.shape[1]} columns and {df.shape[0]} samples")

st.write('\n')

st.subheader('Distributions of Numerical Features')
dis_num_target = st.selectbox(
    'Please select variable to view distribution:',
    numeric_cols)
fig = px.histogram(df, x=dis_num_target, nbins=20)
st.write(fig)

st.write('\n')

st.subheader('Distributions of Categorical Features')
dis_cat_target = st.selectbox(
    'Please select variable to view distribution:',
    categorical_cols)
fig = px.histogram(df, x=dis_cat_target, color=dis_cat_target)
st.write(fig)

st.write('\n')

st.subheader('Correlations')
fig = px.imshow(df.corr())
st.write(fig)

col1, col2 = st.columns(2)
cols_without_target = [x for x in cols_names if x != target]
with col1:
   x_target = st.selectbox(
    'X Axis:',
    cols_without_target
    )

with col2:
    y_target = st.selectbox(
    'Y Axis:',
    cols_without_target
    )

fig = px.scatter(df, x=x_target, y=y_target, color=target)
st.write(fig)

box_num_target = st.selectbox(
    'Please select variable to plot:',
    numeric_cols
    )
fig = px.box(df, x=target, y=box_num_target, color=target)
st.write(fig)
