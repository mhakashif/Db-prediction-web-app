import pickle 
import pandas as pd
import numpy as np
import streamlit as st

loaded_model = pickle.load(open(r'D:/MLD/trained_model.sav', 'rb'))

#creating function
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
     return 'The person is diabetic'
    

def main() :
  st.title('Diabetes prediction Web app')

  #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

  Pregnancies = st.text_input("Number of Pregnancies")
  Glucose = st.text_input("Glucose level")
  BloodPressure = st.text_input("BloodPressure value ")
  SkinThickness = st.text_input("SkinThickness")
  Insulin = st.text_input("Insulin")
  BMI = st.text_input("BMI value")
  DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
  Age = st.text_input("Age of the person")

  #code for prediction
  diagnosis = ''

  #creating button for prediction
  if st.button("Diabetes Test result"):
    diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

  st.success(diagnosis)  


if __name__ == '__main__':
  main()  




