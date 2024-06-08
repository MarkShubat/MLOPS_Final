from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import io
import streamlit as st
import pickle


def load_csv():
    uploaded_file = st.file_uploader( label='Выберите датасет для получения предсказаний')
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        return dataframe
    else:
        return None

@st.cache(allow_output_mutation=True)
def load_model():
    with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
    return pipeline

st.title('Получение предсказаний')
model = load_model()
dataset = load_csv()

result = st.button('Получить предсказание')
if result:
    predictions = model.predict(dataset)
    st.write('**Результат:**')
    st.write(predictions)
