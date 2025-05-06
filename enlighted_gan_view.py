import os

import streamlit as st
import time
import PIL.Image
import numpy as np
from streamlit_image_comparison import image_comparison
import pandas as pd
from enlighten_inference import EnlightenOnnxModel
from metrics.niqe_score import Niqe_score
from metrics.brisque_score import Brisque_score

def process_enlighted_gan(uploaded_file):
    img_pil = PIL.Image.open(uploaded_file)
    img_array = np.array(img_pil)

    model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    processed = model.predict(img_array)  # tidak perlu transpose atau batch dimensi

    return processed

st.markdown("### Low light image enhancement (Dashcam Image) | EnlightedGAN Method")

uploaded_image = st.file_uploader(label="Upload low light image", type=['jpg', 'png'])
if uploaded_image is not None:
    uploaded_image_pil = PIL.Image.open(uploaded_image)
    img_array_uploaded = np.array(uploaded_image_pil)
    with st.spinner("Processing...", show_time=True):
        start_time = time.time()
        enhanced_img = process_enlighted_gan(uploaded_image)
        end_time = time.time()
        execution_time = end_time - start_time
        enhanced_img_pil = PIL.Image.fromarray(enhanced_img)

        niqe = Niqe_score()
        niqe_before = niqe.count(img_array_uploaded)
        niqe_after = niqe.count(enhanced_img)

        brisque = Brisque_score()
        brisque_before = brisque.count(img_array_uploaded)
        brisque_after = brisque.count(enhanced_img)
        # Tampilkan gambar
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption=f"Original image (low light)")
        with col2:
            st.image(enhanced_img, caption=f"Enhanced image")

        st.write(f"Inference time : {execution_time:.3f} s")
        metrics_data = pd.DataFrame({
            "Metrics": ["NIQE", "BRISQUE"],
            "Before": [niqe_before, brisque_before],
            "After": [niqe_after, brisque_after]
        })

        # Tampilkan dengan st.dataframe
        # st.dataframe(metrics_data, use_container_width=True)
        st.dataframe(metrics_data.set_index(metrics_data.columns[0]))