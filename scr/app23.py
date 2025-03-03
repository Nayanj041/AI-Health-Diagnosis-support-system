import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px
from datetime import datetime
import json
import os
from recommendations import get_medical_recommendations

# Set page config at the top of the script
st.set_page_config(
    page_title="Medical Diagnosis Support System",
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
            # (your existing symptoms list)
            'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 
            'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 
            'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',  
            'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 
            'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
            'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 
            'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 
            'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 
            'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
            'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 
            'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 
            'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 
            'malaise', 'blurred_and_distorted_vision', 'phlegm', 
            'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 
            'fast_heart_rate', 'pain_during_bowel_movements', 
            'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
            'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 
            'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
            'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 
            'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
            'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 
            'stiff_neck', 'swelling_joints', 'movement_stiffness', 
            'spinning_movements', 'loss_of_balance', 'unsteadiness', 
            'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
            'continuous_feel_of_urine', 'passage_of_gases', 
            'internal_itching', 'toxic_look_(typhos)', 'depression', 
            'irritability', 'muscle_pain', 'altered_sensorium', 
            'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 
            'watering_from_eyes', 'increased_appetite', 'polyuria', 
            'family_history', 'mucoid_sputum', 'rusty_sputum', 
            'lack_of_concentration', 'visual_disturbances', 
            'receiving_blood_transfusion', 'receiving_unsterile_injections', 
            'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 
            'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 
            'palpitations', 'painful_walking', 'pus_filled_pimples', 
            'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 
            'small_dents_in_nails', 'inflammatory_nails', 'blister', 
            'red_sore_around_nose', 'yellow_crust_ooze'
        ]
        self.conditions = [
            'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 
            'Drug Reaction', 'Peptic ulcer disease', 'AIDS', 'Diabetes', 
            'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 
            'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 
            'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'Hepatitis A', 
            'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 
            'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 
            'Dimorphic hemorrhoids (piles)', 'Heart attack', 'Varicose veins', 
            'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 
            'Osteoarthritis', 'Arthritis', '(vertigo) Paroxysmal Positional Vertigo', 
            'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo'
        ]

    def load_data(self, filename='Training.csv'):
        """Load patient data from a CSV file."""
        try:
            data = pd.read_csv(filename)
            return data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    from sklearn.metrics import confusion_matrix

    def train_model(self):
        """Train the diagnostic models and store confusion matrices."""
        try:
            data = self.load_data()
            if data is None:
                return 0.0, 0.0, 0.0  # Return 0 if data loading fails

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
            return 0.0, 0.0, 0.0

    def diagnose(self, symptoms_dict):
        """Make a diagnosis with confidence scores."""
        try:
            if not hasattr(self.scaler, 'scale_'):
                st.error("Scaler is not fitted. Please train the model first.")
                return None
            
            features = [float(symptoms_dict.get(symptom, 0)) for symptom in self.symptoms_list]
            features_scaled = self.scaler.transform([features])
            
            prediction_rf = self.model_rf.predict(features_scaled)[0]
            probabilities_rf = self.model_rf.predict_proba(features_scaled)[0]

            prediction_dt = self.model_dt.predict(features_scaled)[0]
            probabilities_dt = self.model_dt.predict_proba(features_scaled)[0]



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

            }
        except Exception as e:
            st.error(f"Error in diagnosis: {str(e)}")
            return None

def diagnosis_page():
    st.title("🏥 Advanced Medical Diagnosis Support System")
    st.markdown("---")
    
    if 'diagnosis_system' not in st.session_state:
        diagnosis_system = PatientDiagnosisSystem()
        rf_accuracy, dt_accuracy = diagnosis_system.train_model()
        st.session_state.diagnosis_system = diagnosis_system
        st.session_state.rf_accuracy = rf_accuracy
        st.session_state.dt_accuracy = dt_accuracy

        st.info(f"Models trained successfully. Random Forest Accuracy: {rf_accuracy:.2%}, Decision Tree Accuracy: {dt_accuracy:.2%}")
    
    st.sidebar.header("Patient Information")
    
    patient_info = {
        "name": st.sidebar.text_input("Patient Name", key="patient_name_input"),
        "age": st.sidebar.number_input("Patient Age", 0, 120, 30, key="patient_age_input"),
        "gender": st.sidebar.selectbox("Patient Gender", ["Male", "Female", "Other"], key="patient_gender_input"),
        "medical_history": st.sidebar.text_area("Brief Medical History", key="patient_history_input")
    }
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Symptom Assessment")

        symptoms_dict = {}
        duration_dict = {}  # To store durations for each symptom
        for index, symptom in enumerate(st.session_state.diagnosis_system.symptoms_list):
            symptom_formatted = symptom.replace('_', ' ').title()
            symptoms_dict[symptom] = st.checkbox(symptom_formatted, key=f"symptom_{index}_{symptom}")

        # Symptom duration
        duration = st.slider(f"Days since first symptom for {symptom_formatted}", 1, 14, 1, key=f"symptom_duration_slider_{index}")
        duration_dict[symptom] = duration  # Store the duration for each symptom
        # Initialize diagnosis_result
        diagnosis_result = None

        diagnosis_result = None

        if st.button("Generate Diagnosis", type="primary", key="generate_diagnosis_button"):
            if any(symptoms_dict.values()):
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
                        "duration": duration_dict  # Save the duration dictionary
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
                st.error("Please select at least one symptom.")
    
    with col2:
        if 'diagnosis' in st.session_state:
            st.subheader("Diagnosis Results")
            
            tabs = st.tabs(["Random Forest Diagnosis", "Decision Tree Diagnosis"])
            
            with tabs[0]:
                st.markdown(f"### Primary Diagnosis (Random Forest): **{st.session_state.diagnosis['predicted_condition_rf']}**")
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

                 # Get medical recommendations only if diagnosis_result is available
                if diagnosis_result:
                    recommendations = get_medical_recommendations(diagnosis_result['predicted_condition_dt'])
                    st.markdown(recommendations)


            with tabs[1]:
                st.markdown(f"### Primary Diagnosis (Decision Tree): **{st.session_state.diagnosis['predicted_condition_dt']}**")
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

                 # Get medical recommendations only if diagnosis_result is available
                if diagnosis_result:
                    recommendations = get_medical_recommendations(diagnosis_result['predicted_condition_dt'])
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
    - Machine learning prediction models (Random Forest, Decision Tree, and Logistic Regression)
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