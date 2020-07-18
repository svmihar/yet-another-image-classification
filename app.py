import streamlit as st
import matplotlib.image as mimage
from model import get_prediction  # this will automatically load the model on page load
from model import load_image


def main():
    st.title("shopee test")
    image_file = st.file_uploader("upload an image here", type=["jpg", "jpeg", "png"])
    if image_file is None: 
        st.write('upload image here')
    else: 
        image_tensor, image_pil = load_image(image_file)
        prediction = get_prediction(image_tensor) # TODO: prediction[1] flatten, numpy.array, normalize
        st.write(f"prediction result: {prediction[0]}, prob: {prediction[1]}")
        st.image(image_file, use_column_width=True)


if __name__ == "__main__":
    main()
