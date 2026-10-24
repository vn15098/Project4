import streamlit as st
import tempfile
from pathlib import Path

from search import get_similar_products

st.title(
    "Image-Based Recommendation System"
)

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)

    try:
        st.image(
            uploaded_file,
            caption="Query Image",
            width=250
        )
        with st.spinner("Searching for similar products..."):
            recommendations = get_similar_products(str(temp_path), top_k=5)
    finally:
        temp_path.unlink(missing_ok=True)

    st.subheader("Similar Products")

    cols = st.columns(max(1, min(5, len(recommendations))))

    for col, rec in zip(cols, recommendations):

        with col:
            st.image(rec["path"], use_container_width=True)
            st.caption(f"Cosine similarity: {rec['score']:.3f}")