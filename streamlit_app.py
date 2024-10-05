import streamlit as st

st.title('First APP of Amir')

st.write('Hello world!')

with st.expander('Initial data'):
  st.write('**RAw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

st.write('**X**')
X_raw = df.drop('species', axis=1)
x_raw

st.write('**y**')
y_raw = df.species
y_raw
  
