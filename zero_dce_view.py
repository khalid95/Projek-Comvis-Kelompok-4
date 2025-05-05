import os

import streamlit as st
import time
import PIL.Image
import numpy as np
from streamlit_image_comparison import image_comparison
import pandas as pd
from zero_dce import Trainer
from metrics.niqe_score import Niqe_score
from metrics.brisque_score import Brisque_score

@st.cache_resource
def load_model():
    checkpoint_path = os.path.join("zero_dce_pretrained_models", "model200_dark_faces.pth")

    trainer = Trainer()
    trainer.build_model(pretrain_weights=checkpoint_path)

    return trainer

st.markdown("### Low light image enhancement (Dashcam Image) | Zero DCE Method")

uploaded_image = st.file_uploader(label="Upload low light image", type=['jpg', 'png'])
if uploaded_image is not None:
    uploaded_image_pil = PIL.Image.open(uploaded_image)
    img_array_uploaded = np.array(uploaded_image_pil)
    with st.spinner("Processing...", show_time=True):
        # Load model dan enhance
        trainer = load_model()
        start_time = time.time()
        end_time = time.time()
        execution_time = end_time - start_time
        image, enhanced_img = trainer.infer_cpu(uploaded_image, image_resize_factor=1)

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

        st.write(f"Inferencen time: {execution_time:.3f} detik")

        metrics_data = pd.DataFrame({
            "Metrics": ["NIQE", "BRISQUE"],
            "Before": [niqe_before, brisque_before],
            "After": [niqe_after, brisque_after]
        })
        st.dataframe(metrics_data, use_container_width=True)