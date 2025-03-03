import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Set page config at the top of the script
st.set_page_config(
    page_title="AI Medical Diagnosis Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


class PatientDiagnosisSystem:
    def __init__(self):
        self.model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_dt = DecisionTreeClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.symptoms_list = [
            'fever', 'cough', 'fatigue', 'difficulty_breathing', 'body_ache',
            'headache', 'sore_throat', 'runny_nose', 'nausea', 'diarrhea'
        ]
        self.conditions = [
            'Common Cold', 'Influenza', 'COVID-19', 'Bacterial Infection',
            'Allergic Reaction'
        ]


    def load_data(self, filename='synthetic_patient_data.csv'):
        """Load patient data from a CSV file."""
        try:
            data = pd.read_csv(filename)
            return data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def train_model(self):
        """Train the diagnostic models."""
        try:
            data = self.load_data()
            if data is None:
                return 0.0, 0.0  # Return 0 if data loading fails

            X = data[self.symptoms_list]
            y = data['condition']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train Random Forest
            self.model_rf.fit(X_train_scaled, y_train)
            rf_accuracy = self.model_rf.score(X_test_scaled, y_test)
            st.session_state.rf_cm = confusion_matrix(y_test, self.model_rf.predict(X_test_scaled))

            # Train Decision Tree
            self.model_dt.fit(X_train_scaled, y_train)
            dt_accuracy = self.model_dt.score(X_test_scaled, y_test)
            st.session_state.dt_cm = confusion_matrix(y_test, self.model_dt.predict(X_test_scaled))

            return rf_accuracy, dt_accuracy
        except Exception as e:
            st.error(f"Error in training model: {str(e)}")
            return 0.0, 0.0

    def diagnose(self, symptoms_dict):
        """Make a diagnosis with confidence scores and severity assessment."""
        try:
            features = [float(symptoms_dict.get(symptom, 0)) for symptom in self.symptoms_list]
            features_scaled = self.scaler.transform([features])

            prediction_rf = self.model_rf.predict(features_scaled)[0]
            probabilities_rf = self.model_rf.predict_proba(features_scaled)[0]

            prediction_dt = self.model_dt.predict(features_scaled)[0]
            probabilities_dt = self.model_dt.predict_proba(features_scaled)[0]

            # severity_level = self.calculate_severity(symptoms_dict)

            confidence_scores_rf = {
                condition: float(prob)
                for condition, prob in zip(self.model_rf.classes_, probabilities_rf)
            }

            confidence_scores_dt = {
                condition: float(prob)
                for condition, prob in zip(self.model_dt.classes_, probabilities_dt)
            }

            return {
                'predicted_condition_rf': prediction_rf,
                'confidence_scores_rf': confidence_scores_rf,
                'predicted_condition_dt': prediction_dt,
                'confidence_scores_dt': confidence_scores_dt,
                # 'severity_level': severity_level
            }
        except Exception as e:
            st.error(f"Error in diagnosis: {str(e)}")
            return None

    # def calculate_severity(self, symptoms_dict):
    #     """Calculate severity level based on symptoms."""
    #     severity_weights = {
    #         'fever': 2,
    #         'difficulty_breathing': 3,

    #         # 'fatigue': 1,
    #         # 'rash': 2
    #     }

    #     total_severity = sum(severity_weights.get(symptom, 1)
    #                          for symptom, has_symptom in symptoms_dict.items()
    #                          if has_symptom)

    #     if total_severity <= 4:
    #         return 0  # Mild
    #     elif total_severity <= 8:
    #         return 1  # Moderate
    #     else:
    #         return 2  # Severe


def get_medical_recommendations(diagnosis):
    """Get medical recommendations based on diagnosis"""
    recommendations = {
        "Common Cold": {
            "next_steps": [
                "Rest and get adequate sleep",
                "Stay hydrated with water and warm liquids",
                "Monitor symptoms for 7-10 days"
            ],
            "home_care": [
                "Use over-the-counter cold medications as directed",
                "Gargle with warm salt water for sore throat",
                "Use a humidifier to ease congestion",
                "Take vitamin C supplements"
            ],
            "seek_medical_attention": [
                "Fever above 101.3°F (38.5°C) for more than 3 days",
                "Severe sinus pain or headache",
                "Symptoms lasting more than 10 days",
                "Difficulty breathing or chest pain"
            ]
        },
        "Influenza": {
            "next_steps": [
                "Begin rest immediately",
                "Stay home to prevent spreading",
                "Consider antiviral medications if within 48 hours of symptoms"
            ],
            "home_care": [
                "Take fever reducers (acetaminophen or ibuprofen)",
                "Stay hydrated with water and electrolyte solutions",
                "Use warm compresses for muscle aches",
                "Eat light, nutritious foods"
            ],
            "seek_medical_attention": [
                "Difficulty breathing or shortness of breath",
                "Chest pain or severe abdominal pain",
                "Severe dizziness or confusion",
                "Severe vomiting or inability to keep fluids down"
            ]
        },
        "COVID-19": {
            "next_steps": [
                "Isolate immediately",
                "Get a COVID-19 test",
                "Monitor oxygen levels if possible",
                "Inform close contacts"
            ],
            "home_care": [
                "Rest and stay hydrated",
                "Monitor temperature and symptoms",
                "Take over-the-counter medications for symptoms",
                "Use prone positioning if breathing is difficult"
            ],
            "seek_medical_attention": [
                "Difficulty breathing or shortness of breath",
                "Persistent chest pain or pressure",
                "Confusion or inability to wake/stay awake",
                "Bluish lips or face"
            ]
        },
        "Bacterial Infection": {
            "next_steps": [
                "Schedule a medical appointment",
                "Document all symptoms and their timeline",
                "Monitor temperature regularly"
            ],
            "home_care": [
                "Rest and stay hydrated",
                "Use fever reducers if needed",
                "Monitor symptoms closely",
                "Complete full course of antibiotics if prescribed"
            ],
            "seek_medical_attention": [
                "Fever above 102°F (39°C)",
                "Severe pain or swelling",
                "Symptoms worsen after 48 hours",
                "Development of new symptoms"
            ]
        },
        "Allergic Reaction": {
            "next_steps": [
                "Identify and avoid the trigger if possible",
                "Document symptoms and possible triggers",
                "Consider antihistamines if appropriate"
            ],
            "home_care": [
                "Use air purifiers if indoor allergies",
                "Keep windows closed during high pollen times",
                "Use nasal saline rinses",
                "Take prescribed allergy medications"
            ],
            "seek_medical_attention": [
                "Difficulty breathing or wheezing",
                "Swelling of face, lips, or throat",
                "Dizziness or fainting",
                "Rapid pulse or confusion"
            ]
        }
    }

    recs = recommendations.get(diagnosis, {})

    return f"""
### Recommended Next Steps:
{chr(10).join('- ' + step for step in recs.get('next_steps', []))}

### Home Care Suggestions:
{chr(10).join('- ' + care for care in recs.get('home_care', []))}

### When to Seek Immediate Medical Attention:
{chr(10).join('- ' + warning for warning in recs.get('seek_medical_attention', []))}

---

"""


def diagnosis_page():
    st.title("🏥 Advanced Medical Diagnosis Support System")
    st.markdown("---")

    # Initialize session state
    if 'diagnosis_system' not in st.session_state:
        diagnosis_system = PatientDiagnosisSystem()
        rf_accuracy, dt_accuracy = diagnosis_system.train_model()
        st.session_state.diagnosis_system = diagnosis_system
        st.session_state.rf_accuracy = rf_accuracy
        st.session_state.dt_accuracy = dt_accuracy
        st.info(
            f"Models trained successfully. Random Forest Accuracy: {rf_accuracy:.2%}, Decision Tree Accuracy: {dt_accuracy:.2%}")

    # Initialize diagnosis_result
    diagnosis_result = None

    # Sidebar for patient information
    st.sidebar.header("Patient Information")

    patient_info = {
        "name": st.sidebar.text_input("Patient Name", key="patient_name_input"),
        "age": st.sidebar.number_input("Patient Age", 0, 120, 30, key="patient_age_input"),
        "gender": st.sidebar.selectbox("Patient Gender", ["Male", "Female", "Other"], key="patient_gender_input"),
        "medical_history": st.sidebar.text_area("Brief Medical History", key="patient_history_input")
    }

    # Main content area
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Symptom Assessment")

        # Grouped symptom selection
        st.markdown("### General Symptoms")
        symptoms_dict = {}
        for symptom in st.session_state.diagnosis_system.symptoms_list:
            symptom_formatted = symptom.replace('_', ' ').title()
            symptoms_dict[symptom] = st.checkbox(symptom_formatted, key=f"symptom_{symptom}")

        # Symptom duration
        st.markdown("### Symptom Duration")
        duration = st.slider("Days since first symptom", 1, 14, 1, key="symptom_duration_slider")

        if st.button("Generate Diagnosis", type="primary", key="generate_diagnosis_button"):
            if any(symptoms_dict.values()):
                # Get diagnosis
                diagnosis_result = st.session_state.diagnosis_system.diagnose(symptoms_dict)

                if diagnosis_result:
                    st.session_state.diagnosis = diagnosis_result
                    st.session_state.current_symptoms = symptoms_dict
                    st.session_state.patient_info = patient_info

                    # Save record
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_info": patient_info,
                        "symptoms": symptoms_dict,
                        "diagnosis": diagnosis_result,
                        "duration": duration
                    }

                    try:
                        filename = "patient_records.json"
                        if os.path.exists(filename):
                            with open(filename, 'r') as f:
                                records = json.load(f)
                        else:
                            records = []

                        records.append(record)

                        with open(filename, 'w') as f:
                            json.dump(records, f, indent=4)
                    except Exception as e:
                        st.warning(f"Could not save patient record: {str(e)}")

                else:
                    st.error("Diagnosis could not be generated.")
            else:
                st.error("Please select at least one symptom.")

    with col2:
        if 'diagnosis' in st.session_state:
            st.subheader("Diagnosis Results")

            tabs = st.tabs(["Random Forest Diagnosis", "Decision Tree Diagnosis"])

            with tabs[0]:
                st.markdown(
                    f"### Primary Diagnosis (Random Forest): **{st.session_state.diagnosis['predicted_condition_rf']}**")
                st.write("### Confidence Scores:")
                confidence_scores_rf = st.session_state.diagnosis['confidence_scores_rf']
                confidence_df_rf = pd.DataFrame.from_dict(
                    confidence_scores_rf, orient='index', columns=['Probability']
                ).sort_values('Probability', ascending=False)

                fig_confidence_rf = px.bar(
                    confidence_df_rf,
                    x=confidence_df_rf.index,
                    y='Probability',
                    title='Random Forest Condition Probability',
                    labels={'Probability': 'Confidence'}
                )
                st.plotly_chart(fig_confidence_rf)

            with tabs[1]:
                st.markdown(
                    f"### Primary Diagnosis (Decision Tree): **{st.session_state.diagnosis['predicted_condition_dt']}**")
                st.write("### Confidence Scores:")
                confidence_scores_dt = st.session_state.diagnosis['confidence_scores_dt']
                confidence_df_dt = pd.DataFrame.from_dict(
                    confidence_scores_dt, orient='index', columns=['Probability']
                ).sort_values('Probability', ascending=False)

                fig_confidence_dt = px.bar(
                    confidence_df_dt,
                    x=confidence_df_dt.index,
                    y='Probability',
                    title='Decision Tree Condition Probability',
                    labels={'Probability': 'Confidence'}
                )
                st.plotly_chart(fig_confidence_dt)

            if diagnosis_result:
                recommendations = get_medical_recommendations(diagnosis_result['predicted_condition_rf'])
                st.markdown(recommendations)


def comparison_page():
    """Comparison page for model performance evaluation."""
    st.title("📊 Model Comparison")
    st.markdown("---")

    if 'diagnosis_system' in st.session_state:
        st.subheader("Model Performance Metrics")
        metrics_data = {
            "Model": ["Random Forest", "Decision Tree"],
            "Accuracy": [st.session_state.rf_accuracy, st.session_state.dt_accuracy]
        }
        metrics_df = pd.DataFrame(metrics_data)

        st.dataframe(metrics_df)

        # Plotting the accuracies
        fig = px.bar(metrics_df, x='Model', y='Accuracy', title='Model Accuracy Comparison',
                     labels={'Accuracy': 'Accuracy (%)'})
        st.plotly_chart(fig)

        st.subheader("Confusion Matrices")
        st.write("#### Random Forest Confusion Matrix:")
        st.write(st.session_state.rf_cm)

        st.write("#### Decision Tree Confusion Matrix:")
        st.write(st.session_state.dt_cm)

def record_management_page():
    """Additional page for record management"""
    st.title("🗄️ Patient Record Management")

    records = load_records()

    if not records:
        st.info("No patient records found.")
        return

    st.write(f"Total Records: {len(records)}")

    # Display records in a table
    record_data = []
    for record in records:
        # Use .get() to avoid KeyError if the key doesn't exist
        patient_info = record.get('patient_info', {})
        diagnosis = record.get('diagnosis', {}).get('predicted_condition_rf', 'N/A')
        name = patient_info.get('name', 'Unknown')

        record_data.append({
            "Name": name,
            "Diagnosis": diagnosis,
            "Date": record.get('timestamp', 'Unknown Date')
        })

    records_df = pd.DataFrame(record_data)
    st.dataframe(records_df)


def load_records():
    """Utility function to load patient records"""
    try:
        filename = "patient_records.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading records: {str(e)}")
        return []


def about_page():
    """About page for the Medical Diagnosis Support System"""
    st.title("🩺 About Medical Diagnosis Support System")

    st.markdown("""
    ### Overview
    This AI-powered Medical Diagnosis Support System is designed to assist healthcare professionals 
    and individuals in preliminary medical assessments.

    ### Key Features
    - Symptom-based diagnosis
    - Machine learning prediction models (Random Forest and Decision Tree)
    - Personalized recommendations
    - PDF report generation

    ### Disclaimer
    This system is an AI tool and should NOT replace professional medical consultation. 
    Always seek advice from qualified healthcare providers.

    ### Technology Stack
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    """)


def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page",
                                ["Diagnosis", "Record Management", "Model Comparison", "About"],
                                key="navigation_radio"  # Ensure this key is unique
                                )

    if app_mode == "Diagnosis":
        diagnosis_page()
    elif app_mode == "Record Management":
        record_management_page()
    elif app_mode == "Model Comparison":
        comparison_page()
    elif app_mode == "About":
        about_page()


if __name__ == "__main__":
    main()