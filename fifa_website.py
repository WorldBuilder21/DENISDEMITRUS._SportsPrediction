import streamlit as stl
import pandas as pd
import pickle as pkl


best_variables = ['value_eur',
 'movement_reactions',
 'age',
 'wage_eur',
 'potential',
 'international_reputation',
 'release_clause_eur']

with open("RandomForestRegressor.pkl", "rb") as f:
    preparedModel = pkl.load(f)

stl.title("⚽ Fifa overall score predictor")
stl.markdown("A simple web app used to predict the overall score of fifa players, picking top 7 varibales which had the highest correlation during training of the model. The model bost of an r2 score of 0.9839.")
stl.markdown(":red[Built by Denis Demitrus]")

user_inputs = {}
for x in best_variables:
    user_inputs[x] = stl.number_input(f"{x}", value=0)

if stl.button('Predict players overall rating'):
    df = pd.DataFrame([user_inputs])
    score = preparedModel.predict(df)
    stl.write(f':green[Predicted rating: {score[0]}]')
