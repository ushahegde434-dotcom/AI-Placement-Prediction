
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 100

cgpa_arr = np.random.uniform(5.0, 10.0, n)
internships_arr = np.random.randint(0, 4, n)
projects_arr = np.random.randint(0, 6, n)
coding_skill_arr = np.random.randint(30, 100, n)

placed = []
for i in range(n):
    score = cgpa_arr[i]*10 + internships_arr[i]*10 + projects_arr[i]*5 + coding_skill_arr[i]
    placed.append(1 if score > 140 else 0)

df = pd.DataFrame({
    'CGPA': cgpa_arr.round(2),
    'Internships': internships_arr,
    'Projects': projects_arr,
    'Coding_Skill': coding_skill_arr,
    'Placed': placed
})

X = df[['CGPA','Internships','Projects','Coding_Skill']]
y = df['Placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.title('AI Placement Prediction System')
st.write(f'Model Accuracy: {accuracy*100:.2f}%')

cgpa = st.number_input('CGPA', 0.0, 10.0, 8.0)
internships = st.number_input('Internships', 0, 10, 2)
projects = st.number_input('Projects', 0, 10, 3)
coding_skill = st.slider('Coding Skill', 0, 100, 80)

if st.button('Predict Placement'):
    prediction = model.predict([[cgpa, internships, projects, coding_skill]])[0]
    probability = model.predict_proba([[cgpa, internships, projects, coding_skill]])[0][1] * 100

    if prediction == 1:
        st.success('Likely to Get Placed ✅')
    else:
        st.error('Placement Chances are Low ❌')

    st.write(f'Placement Probability: {probability:.2f}%')

st.subheader('Sample Dataset')
st.dataframe(df.head(20))
