import streamlit as st

from src.predictor import train_model


st.title("Student Performance Predictor")

st.write(
    "Enter some student information to predict "
    "the expected result."
)


study_hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=12.0,
    value=3.0
)

attendance = st.number_input(
    "Attendance percentage",
    min_value=0,
    max_value=100,
    value=75
)

assignments = st.number_input(
    "Assignments completed",
    min_value=0,
    max_value=10,
    value=6
)

previous_score = st.number_input(
    "Previous exam score",
    min_value=0,
    max_value=100,
    value=60
)


if st.button("Predict Result"):

    model, accuracy = train_model()

    prediction = model.predict([
        [
            study_hours,
            attendance,
            assignments,
            previous_score
        ]
    ])

    st.subheader("Prediction")

    st.write(prediction[0])

    st.write(
        "Model accuracy on test data:",
        round(accuracy * 100, 2),
        "%"
    )
