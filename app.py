import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import streamlit as st
from catboost import CatBoostRegressor


def show_title_with_subtitle():
    # Заголовок и подзаголовок
    st.title("Цены на жилье")
    st.write("# Продвинутые методы регрессии")


def show_info_page():
    st.write("### Задача")
    st.write(
        "Построить, обучить и оценить модель для решения задачи регрессии - получения высокоточных предсказаний стоимости"
        "жилого дома по совокупности множества описывающих признаков, влияющих на конечную стоимость.")
    st.image("https://www.propertyreporter.co.uk/images/660x350/16402-shutterstock_538341163.jpg",
             use_column_width=True)
    st.write("### Описание входных данных")
    st.write(
        "Данные, для которых необходимо получать предсказания, представляют собой подробное признаковое описание жилых домов,"
        "включающее в себя такие факторы, как район расположения, особенности архитектуры здания, качественная и количественная"
        "оценки функциональных возможностей жилого дома (отопление, электричество, качество прилегающей территории, пешеходных и "
        "автомобильных дорог) и другие.")
    st.write("### Выбранная регрессионная модель")
    st.write(
        "В результате анализа метрик качества нескольких продвинутых регрессионных композиционных моделей выбрана модель"
        "CatBoostRegressor (https://catboost.ai/docs/concepts/python-reference_catboostregressor.html), обеспечивающая"
        "высокое качество предсказаний стоимости жилых домов.")
    st.write("Более подробно с библиотека catboost представлена в предложенном видео:")
    st.video("https://www.youtube.com/watch?v=UYDwhuyWYSo")
    st.write("Выполненная работа представляет собой результат участия в соревновании на портале Kaggle. Более подробно"
             "ознакомиться с правилами соревнования можно по ссылке ниже:")
    st.write("https://www.kaggle.com/c/house-prices-advanced-regression-techniques")


def show_predictions_page():
    st.write("Файл для примера: https://drive.google.com/file/d/1z-_CFKFgS5-cf3pekPKtwogadBQeUWLe/view?usp=sharing")
    file = st.file_uploader(label="Выберите csv файл с предобработанными данными для прогнозирования стоимости",
                            type=["csv"],
                            accept_multiple_files=False)
    if file is not None:
        test_data = pd.read_csv(file)
        st.write("### Загруженные данные")
        st.write(test_data)
        make_predictions(get_model(), test_data)


def get_model():
    return CatBoostRegressor().load_model(os.path.join(os.path.dirname(__file__), "models", "cb"))


def make_predictions(model, X):
    st.write("### Предсказанные значения")
    pred = pd.DataFrame(model.predict(X))
    st.write(pred)
    st.write("### Гистограмма распределения предсказаний")
    plot_hist(pred)


def plot_hist(data):
    fig = plt.figure()
    sbn.histplot(data, legend=False)
    st.pyplot(fig)


def select_page():
    # Сайдбар для смены страницы
    return st.sidebar.selectbox("Выберите страницу", ("Информация", "Прогнозирование"))


# Стиль для скрытия со страницы меню, футера streamlit и кнопки deploy
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# размещение элементов на странице
show_title_with_subtitle()
st.sidebar.title("Меню")
page = select_page()
st.sidebar.write("© Mikhail Gunenkov 2024")

if page == "Информация":
    show_info_page()
else:
    show_predictions_page()
