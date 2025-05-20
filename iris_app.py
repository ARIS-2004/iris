import streamlit as st
import pandas as pd
from iris_model import train_model

model, iris = train_model()

st.set_page_config(page_title="Iris Flower Classifier 🌸", layout="centered", initial_sidebar_state="expanded")

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)

# Custom styled title replacing st.title to reduce padding
st.markdown(
    "<h1 style='padding: 4px 0; margin: 0; text-align:center;'>🌸 Iris Flower Classification</h1>",
    unsafe_allow_html=True
)

st.markdown("Adjust the sliders on the left to enter measurements.", unsafe_allow_html=True)

prediction = model.predict(input_data)[0]
species_name = iris.target_names[prediction].capitalize()

color_map = {
    'Setosa': '#FF6F61',
    'Versicolor': '#6B5B95',
    'Virginica': '#88B04B'
}
color = color_map.get(species_name, '#000000')

image_map = {
    'Setosa': "https://live.staticflickr.com/65535/51376589362_b92e27ae7a_z.jpg",
    'Versicolor': "https://th.bing.com/th/id/OIP.pY8BlneJNNK9ISQ_CnfGCAHaFj?cb=iwc2&rs=1&pid=ImgDetMain",
    'Virginica': "https://th.bing.com/th/id/OIP.XYHIv32qs7zqokUohFTCawHaJ4?cb=iwc2&rs=1&pid=ImgDetMain"
}
image_url = image_map.get(species_name, "")

st.markdown(
    f'<h2 style="color:{color}; text-align:center; padding: 2px 0; margin: 2px 0;">Predicted species: <strong>{species_name}</strong></h2>',
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='margin: 2px 4px 4px 4px; font-weight:600;'>Input measurements:</p>"
    f"<p style='margin: 0;'>Sepal Length: {sepal_length} cm | Sepal Width: {sepal_width} cm<br>"
    f"Petal Length: {petal_length} cm | Petal Width: {petal_width} cm</p>",
    unsafe_allow_html=True
)

if image_url:
    st.image(image_url, caption=f"Iris {species_name}", width=200)
