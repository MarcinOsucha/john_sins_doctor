# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "new_model.sv"
model = pickle.load(open(filename, "rb"))
# otwieramy wcześniej wytrenowany model

symptoms = {1: "Bezsenność", 2: "Ogólne zmęczenie", 3: "Złe samopoczucie", 4: "Słaba kondycja", 5: "Duszności"}
medicaments = {1: "Aviomarin", 2: "LSD", 3: "Nervosol", 4: "Chleb"}


def main():

    st.set_page_config(page_title="Zaburzenia rytmu serca")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://i0.wp.com/www.opindia.com/wp-content/uploads/2021/03/johnny-sins.jpg?fit=933%2C521&ssl=1")

    with overview:
        st.title("Czy masz zaburzenia rytmu serca")

    with left:
        medicaments_radio = st.radio("Jakie przyjmujesz leki", list(medicaments.keys()), format_func=lambda x: medicaments[x])
        symptoms_radio = st.radio("Jakie masz objawy", list(symptoms.keys()), index=2, format_func=lambda x: symptoms[x], )

    with right:
        age_slider = st.slider("Wiek", value=1, min_value=1, max_value=90)
        height_slider = st.slider(
            "Wzrost", min_value=150, max_value=215
        )
        diseases = st.slider(
            "Ilość innych chorób", min_value=0, max_value=5
        )

    data = [
        [
            symptoms_radio,
            medicaments_radio,
            age_slider,
            height_slider,
            diseases,
        ]
    ]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba zachoruje na przewlekłe zapalenie płata czołowego nadgarstka")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write(
            "Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100)
        )


if __name__ == "__main__":
    main()
