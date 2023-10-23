import streamlit as st
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
    potential = st.number_input("Potential:")
    wage_eur = st.number_input("Wage (EUR):")
    age = st.number_input("Age:")
    international_reputation = st.number_input("International Reputation:")
    release_clause_eur = st.number_input("Release Clause (EUR):")
    shooting = st.number_input("Shooting:")
    passing = st.number_input("Passing:")
    dribbling = st.number_input("Dribbling:")
    physic = st.number_input("Physic:")
    attacking_crossing = st.number_input("Attacking Crossing:")
    attacking_short_passing = st.number_input("Attacking Short Passing:")
    skill_curve = st.number_input("Skill Curve:")
    skill_long_passing = st.number_input("Skill Long Passing:")
    skill_ball_control = st.number_input("Skill Ball Control:")
    movement_reactions = st.number_input("Movement Reactions:")
    power_shot_power = st.number_input("Power Shot Power:")
    power_long_shots = st.number_input("Power Long Shots:")
    mentality_aggression = st.number_input("Mentality Aggression:")
    mentality_vision = st.number_input("Mentality Vision:")
    mentality_composure = st.number_input("Mentality Composure:")

    if st.button("Predict"):
        # Make predictions based on user input
        user_input = np.array([
            potential, wage_eur, age, international_reputation,
            release_clause_eur, shooting, passing, dribbling, physic,
            attacking_crossing, attacking_short_passing, skill_curve,
            skill_long_passing, skill_ball_control, movement_reactions,
            power_shot_power, power_long_shots, mentality_aggression,
            mentality_vision, mentality_composure
        ]).reshape(1,-1)

        input_features=scaler.transform(user_input)
        
        prediction = best_model_loaded.predict(input_features)
        
        # Display the prediction
        st.write(f"Predicted Value: {prediction[0]} EUR")

if __name__ == "__main__":
    main()

