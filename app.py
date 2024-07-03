import streamlit as st
import pickle
import numpy as np

# Load the trained model from a pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict the output based on user input
def predict_output(input_values):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit app
st.title('Machine Learning Model Prediction')

# Take user input
st.header('Enter the input values:')
input1 = st.number_input('Mill Speed (RPM)', value=0.0)
input2 = st.number_input('Inlet Feed Size (mm)', value=0.0)
input3 = st.number_input('Inlet Air Flow (m3/min)', value=0.0)
input4 = st.number_input('Grinding Media Size (mm)', value=0.0)
input5 = st.number_input('Grinding Pressure(Mpa)', value=0.0)

# Collect inputs in a list
input_values = [input1, input2, input3, input4, input5]

# Predict and display the result
if st.button('Predict'):
    prediction = predict_output(input_values)
    st.write(f'The predicted output is: {prediction}')
