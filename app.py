import streamlit as st
import pickle
import numpy as np
import time

st.title('Calorie Burn Predictor')
st.markdown('##### Estimate how many calories you burn during a workout.')

model = pickle.load(open('regressor.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
poly = pickle.load(open('poly_reg.pkl','rb'))

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Personal Details")

    with st.container():
        gender = st.radio(
            "Gender",
            ["male", "female"],
            horizontal=True
        )

    with st.container():
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=80,
            value=25
        )

    with st.container():
        height = st.number_input(
            "Height (cm)",
            min_value=120,
            max_value=220,
            value=170
        )

    with st.container():
        weight = st.number_input(
            "Weight (kg)",
            min_value=30,
            max_value=150,
            value=70
        )


with col2:
    st.subheader("Workout Details")

    with st.container():
        duration = st.number_input(
            "Duration of workout (minutes)",
            min_value=1,
            max_value=300,
            value=30
        )

    with st.container():
        heart_rate = st.number_input(
            "Average Heart Rate (bpm)",
            min_value=60,
            max_value=200,
            value=120
        )

    with st.container():
        body_temp = st.number_input(
            "Body Temperature (°C)",
            min_value=35.0,
            max_value=40.0,
            value=37.0,
            step=0.1
        )


st.divider()
if st.button('Predict Calories Burned'):
    X = [[gender,age,height,weight,duration,heart_rate,body_temp]]
    X = np.array(encoder.transform(X))
    X_poly = poly.transform(X)
    predicted = model.predict(X_poly)
    with st.spinner('Calculating your burn....'):
        time.sleep(1)
    st.markdown('### Estimated Calories Burned')
    st.metric('',predicted.round(2))
    if predicted.round(2) < 200:
        st.markdown(' This was a **light** workout. Great for **recovery** or **beginners**')
    elif predicted.round(2) < 500:
        st.markdown(' **Nice!** This was a moderate workout with **good** calorie burn')
    else:
        st.markdown(' **Intense** workout! Make sure to stay hydrated and rest well')