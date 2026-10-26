import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# ------------------------------------
# APPLICATION CONFIGURATION
# ------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "student_risk_model.joblib"


# ------------------------------------
# LOAD THE TRAINED MODEL
# ------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error(
        "Unable to load the trained model. "
        "Check that student_risk_model.joblib exists "
        "and that your package versions are compatible."
    )

    st.stop()


# ------------------------------------
# PAGE HEADER
# ------------------------------------

st.title(
    "🎓 AI-Based Student Performance Prediction "
    "& Early Support System"
)

st.write(
    "Estimate student academic risk from early-course "
    "information and identify students who may benefit "
    "from additional support."
)

st.info(
    "This tool provides estimates, not certain outcomes. "
    "Predictions should be reviewed by an educator and "
    "must not be used as the sole basis for consequential "
    "decisions about a student."
)


# ------------------------------------
# SIDEBAR
# ------------------------------------

st.sidebar.header("About the Project")

st.sidebar.write(
    "This application uses a machine learning model "
    "trained on historical student data."
)

st.sidebar.write(
    "The prediction target is the historical binary "
    "at-risk label."
)


# ------------------------------------
# FILE UPLOAD
# ------------------------------------

st.header("Upload Student Data")

st.write(
    "Upload a CSV containing the model's input features. "
    "Do not include final_result or at_risk as input features."
)

uploaded_file = st.file_uploader(
    "Choose a student CSV file",
    type=["csv"]
)


# ------------------------------------
# RECOMMENDATION FUNCTION
# ------------------------------------

def recommend_support(row, risk_probability):

    recommendations = []

    missed = row.get(
        "missed_assessments_by_day30",
        0
    )

    active_days = row.get(
        "vle_active_days_first30",
        0
    )

    score = row.get(
        "early_assessment_mean_score",
        None
    )

    if pd.notna(missed) and missed > 0:

        recommendations.append(
            "Review missed assessments and offer a "
            "manageable catch-up plan."
        )

    if pd.notna(active_days) and active_days < 5:

        recommendations.append(
            "Check whether the student needs help "
            "accessing or using learning resources."
        )

    if pd.isna(score):

        recommendations.append(
            "Check whether an early assessment was "
            "completed and whether assistance is needed."
        )

    if risk_probability >= 0.5:

        recommendations.append(
            "Offer a supportive check-in to understand "
            "the student's needs and discuss next steps."
        )

    else:

        recommendations.append(
            "Continue routine monitoring and make "
            "academic support resources available."
        )

    return recommendations


# ------------------------------------
# PREDICTION
# ------------------------------------

if uploaded_file is not None:

    try:

        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            input_df.head(10),
            use_container_width=True
        )

        # Recover the original training feature names
        # from the saved preprocessing pipeline.

        preprocessor = model.named_steps["preprocessor"]

        numeric_columns = list(
            preprocessor.transformers_[0][2]
        )

        categorical_columns = list(
            preprocessor.transformers_[1][2]
        )

        required_columns = (
            numeric_columns + categorical_columns
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in input_df.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded file is missing required "
                "columns: "
                + ", ".join(missing_columns)
            )

            st.stop()

        # Use only the features expected by the model.
        # Extra columns are ignored.

        X_new = input_df[required_columns].copy()

        # Generate predictions.

        predictions = model.predict(X_new)

        probabilities = model.predict_proba(X_new)[:, 1]

        results = input_df.copy()

        results["predicted_at_risk"] = predictions

        results["risk_probability_percent"] = (
            probabilities * 100
        ).round(2)

        results["predicted_category"] = [
            "At Risk" if prediction == 1
            else "Low Risk"
            for prediction in predictions
        ]

        st.subheader("Prediction Results")

        total_records = len(results)

        at_risk_count = int(
            (predictions == 1).sum()
        )

        low_risk_count = total_records - at_risk_count

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Records Analyzed",
            total_records
        )

        col2.metric(
            "Flagged as At Risk",
            at_risk_count
        )

        col3.metric(
            "Flagged as Low Risk",
            low_risk_count
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        # ------------------------------------
        # FILTER AT-RISK RECORDS
        # ------------------------------------

        st.subheader("Students Flagged for Review")

        at_risk_results = results[
            results["predicted_at_risk"] == 1
        ]

        st.dataframe(
            at_risk_results,
            use_container_width=True
        )

        # ------------------------------------
        # INDIVIDUAL SUPPORT RECOMMENDATIONS
        # ------------------------------------

        st.subheader("Early-Support Recommendations")

        if not results.empty:

            selected_index = st.selectbox(
                "Select a record",
                options=list(results.index),
                format_func=lambda index: (
                    f"Record {index + 1}"
                )
            )

            selected_row = X_new.loc[selected_index]

            selected_probability = probabilities[
                list(results.index).index(selected_index)
            ]

            selected_prediction = predictions[
                list(results.index).index(selected_index)
            ]

            if selected_prediction == 1:

                st.warning(
                    "The model flags this record for review."
                )

            else:

                st.success(
                    "The model does not flag this record "
                    "as at risk at the current threshold."
                )

            st.write(
                "Estimated risk probability:",
                f"{selected_probability * 100:.2f}%"
            )

            st.write(
                "This probability is an estimate and "
                "should not be treated as a certainty."
            )

            support_actions = recommend_support(
                selected_row,
                selected_probability
            )

            for action in support_actions:

                st.write(f"- {action}")

        # ------------------------------------
        # DOWNLOAD RESULTS
        # ------------------------------------

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="student_risk_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:

        st.error(
            "An error occurred while processing the file. "
            "Check the input format and model compatibility."
        )

        st.exception(error)

else:

    st.info(
        "Upload a CSV containing the required input "
        "features to generate predictions."
    )


# ------------------------------------
# FOOTER
# ------------------------------------

st.divider()

st.caption(
    "Academic project demonstration. Predictions are "
    "decision-support estimates, not definitive "
    "judgments about students."
)