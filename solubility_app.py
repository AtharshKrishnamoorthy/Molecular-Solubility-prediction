import numpy as np
import pandas as pd
import streamlit as st
import pickle
import altair as alt
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn import *

# Custom function to calculate aromatic proportion
def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = [1 for i in aromatic_atoms if i]
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom / HeavyAtom
    return AR

# Function to generate molecular descriptors
def generate(smiles, verbose=False):
    moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
    baseData = np.arange(1, 1)
    for i, mol in enumerate(moldata):
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)
        row = np.array([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds, desc_AromaticProportion])
        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    return descriptors

# Page Title
st.write("""
# Molecular Solubility Prediction Web App
""")

st.info("""
Solubility is the ability of a substance, called the solute, to dissolve in another substance, called the solvent, to form a solution. The solute can be a solid, liquid, or gas, while the solvent is usually a liquid or solid.

This app predicts the **Solubility (LogS)** values of molecules!
""")

image = Image.open('D:\Projects\Molecular Solubility Prediction\Screenshot 2024-05-18 170832.png')
st.image(image, use_column_width=True)

# Input molecules (Side Panel)
st.sidebar.header('User Input Features')

# Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"
SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES # Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
st.write(SMILES[1:]) # Skips the dummy first item

# Calculate molecular descriptors
st.header('Computed molecular descriptors')
X = generate(SMILES)
X = X[1:] # Skips the dummy first item
st.write(X)

with open('solubility_model.pkl','rb') as file:
    model_loaded = pickle.load(file)

# Apply model to make predictions
prediction = model_loaded.predict(X)

st.header('Predicted LogS values')
st.write(prediction) # Display all predicted values

# Analysis and Visualizations
st.header('Analysis and Visualizations')

# Classification
solubility_classes = pd.cut(prediction, bins=[-np.inf, -1, 0, np.inf], labels=["Poorly Soluble", "Moderately Soluble", "Highly Soluble"])
X['LogS'] = prediction
X['Solubility Class'] = solubility_classes

# Histogram of predicted LogS values using Streamlit's chart
st.subheader('Histogram of Predicted LogS Values')
st.write("""
The histogram is a graphical representation that organizes a group of data points into user-specified ranges. It is a useful tool for understanding the distribution of predicted LogS values, showing how frequently each range of values occurs in the data set.
""")
st.bar_chart(np.histogram(prediction, bins=20)[0])

# Scatter plot of LogS vs. Molecular Weight using Altair
st.subheader('LogS vs. Molecular Weight')
st.write("""
The scatter plot is a type of data visualization that shows the relationship between two variables. In this plot, we can see the predicted LogS values against the molecular weight of the compounds. This helps in identifying any correlations or patterns between molecular weight and solubility.
""")

scatter_plot = alt.Chart(X).mark_circle(size=60).encode(
    x='MolWt',
    y='LogS',
    color='Solubility Class',
    tooltip=['MolWt', 'LogS', 'Solubility Class']
).interactive()

st.altair_chart(scatter_plot, use_container_width=True)

# Display solubility class with text color
st.subheader('Solubility Class')
for i, sol_class in enumerate(solubility_classes):
    color = 'red' if sol_class == "Poorly Soluble" else 'blue' if sol_class == "Moderately Soluble" else 'green'
    st.markdown(f"<span style='color: {color};'>Compound {i+1} is {sol_class}</span>", unsafe_allow_html=True)

# Allow user to input custom data for prediction using sliders
st.sidebar.header('Predict LogS for Custom Input')

# Sliders for each descriptor
MolLogP = st.sidebar.slider('MolLogP', min_value=-5.0, max_value=5.0, value=2.5, step=0.1)
MolWt = st.sidebar.slider('MolWt', min_value=0.0, max_value=1000.0, value=150.34, step=1.0)
NumRotatableBonds = st.sidebar.slider('NumRotatableBonds', min_value=0, max_value=20, value=1, step=1)
AromaticProportion = st.sidebar.slider('AromaticProportion', min_value=0.0, max_value=1.0, value=0.0, step=0.01)

# Collecting the custom input data from sliders
custom_input_data = np.array([[MolLogP, MolWt, NumRotatableBonds, AromaticProportion]])

# Predict button to generate the prediction
if st.sidebar.button('Predict'):
    custom_prediction = model_loaded.predict(custom_input_data)
    st.sidebar.write('Predicted LogS for custom input:', custom_prediction[0])
    custom_solubility_class = pd.cut(custom_prediction, bins=[-np.inf, -1, 0, np.inf], labels=["Poorly Soluble", "Moderately Soluble", "Highly Soluble"])[0]
    custom_color = 'red' if custom_solubility_class == "Poorly Soluble" else 'blue' if custom_solubility_class == "Moderately Soluble" else 'green'
    st.sidebar.markdown(f"<span style='color: {custom_color};'>The compound is {custom_solubility_class}</span>", unsafe_allow_html=True)
