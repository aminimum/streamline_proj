import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Titanic Machine Learning From Disaster")

data = pd.read_csv("data/train.csv")
st.write(data.head())

features = data[["Pclass", "Age", "SibSp", "Parch"]]

target = data["Survived"]

features = features.fillna(0)


model = RandomForestClassifier()

model.fit(features, target)

st.subheader("Введите данные для пассажира:")
pclass = st.selectbox("Класс пасcажира:", sorted(data["Pclass"].unique()))
age = st.selectbox("Возраст", sorted(data["Age"].unique()))
sibsp = st.selectbox("Количество братьев/сестер:", sorted(data["SibSp"].unique()))
parch = st.selectbox("Количество детей:", sorted(data["Parch"].unique()))

user_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
     "SibSp": [sibsp],
     "Parch": [parch]
})

prediction = model.predict(user_data)

if prediction:
    st.success("Пассажир скорее всего выжил")
else:
    st.error("Пассажир скорее всего погиб")
    
# st.title("Visualiztion")
# x = np.linspace(0, 1, 100)
# y = np.sin(x)

# fig, ax = plt.subplots()
# ax.plot(x, y)

# st.pyplot(fig)





# option = st.sidebar.selectbox("Choose section", ("Home Page", "Settings", "Portal"))

# if option=="Home Page":
#     st.title("Home Page")
# elif option=="Settings":
#     st.title("Settings")
# else:
#     st.title("Portal")

