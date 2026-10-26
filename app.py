```python
import streamlit as st
import joblib
import sklearn
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "student_risk_model.joblib"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error("Unable to load the trained model.")

    st.write("Expected model path:", str(MODEL_PATH))
    st.write("Model file exists:", MODEL_PATH.is_file())
    st.write("Scikit-learn version:", sklearn.__version__)

    st.exception(error)
    st.stop()
```
