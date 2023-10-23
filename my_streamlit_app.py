import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
model_file_path = 'C:\\Users\\hanna\\Downloads\\deployment\\best_model.plk'
sc= 'C:\\Users\\hanna\\Downloads\\deployment\\scaler.joblib'
from joblib import load
# Load the pickled model
with open(model_file_path, 'rb') as f:
    best_model_loaded = pickle.load(f)

scaler = load(sc)

st.title("Machine Learning Model for player prediction")

def main():
    value_eur = st.number_input("Value (EUR):")
    release_clause_eur = st.number_input("Release Clause (EUR):")
    age = st.number_input("Age:")
    potential = st.number_input("Potential:")
    movement_reactions = st.number_input("Movement Reactions:")
    wage_eur = st.number_input("Wage (EUR):")
    defending = st.number_input("Defending:")
    club_name = st.text_input("Club Name:")
    mentality_interceptions = st.number_input("Mentality Interceptions:")
    league_name = st.text_input("League Name:")
    attacking_crossing = st.number_input("Attacking Crossing:")
    goalkeeping_diving = st.number_input("Goalkeeping Diving:")
    mentality_composure = st.number_input("Mentality Composure:")
    goalkeeping_reflexes = st.number_input("Goalkeeping Reflexes:")
    defending_marking_awareness = st.number_input("Defending Marking Awareness:")
    goalkeeping_positioning = st.number_input("Goalkeeping Positioning:")
    defending_standing_tackle = st.number_input("Defending Standing Tackle:")
    mentality_penalties = st.number_input("Mentality Penalties:")

    if st.button("Predict"):
        # Make predictions based on user input
        user_input = np.array([
    value_eur, release_clause_eur, age, potential, movement_reactions, wage_eur,
    defending, club_name,mentality_interceptions,league_name, attacking_crossing, goalkeeping_diving,
    mentality_composure, goalkeeping_reflexes, defending_marking_awareness,
    goalkeeping_positioning, defending_standing_tackle, mentality_penalties
]).reshape(1,-1)

        input_features=scaler.transform(user_input)

# Encode categorical variables
        label_encoder = LabelEncoder()
        user_input['club_name'] = label_encoder.fit_transform(user_input['club_name'])
        user_input['league_name'] = label_encoder.fit_transform(user_input['league_name'])
        input_features=scaler.transform(user_input)
        prediction = best_model_loaded.predict(input_features)
        
        # Display the prediction
        st.write(f"Predicted Value: {prediction[0]} EUR")

if __name__ == "__main__":
    main()

