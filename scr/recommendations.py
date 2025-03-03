def get_medical_recommendations(diagnosis):
    """Get medical recommendations based on diagnosis"""
    recommendations = {
        'Fungal infection': {
            "next_steps": [
                "Rest and get adequate sleep",
                "Stay hydrated with water and warm liquids",
                "Monitor symptoms for 7-10 days"
            ],
            "home_care": [
                "Use over-the-counter antifungal creams as directed",
                "Keep the affected area clean and dry",
                "Wear loose-fitting clothing"
            ],
            "seek_medical_attention": [
                "Fever above 101.3°F (38.5°C) for more than 3 days",
                "Severe symptoms or worsening condition"
            ]
        },
        'Allergy': {
            "next_steps": [
                "Identify and avoid allergens",
                "Consider taking antihistamines",
                "Monitor symptoms closely"
            ],
            "home_care": [
                "Use saline nasal sprays for nasal congestion",
                "Apply cool compresses for skin reactions"
            ],
            "seek_medical_attention": [
                "Severe allergic reactions (anaphylaxis)",
                "Difficulty breathing or swallowing"
            ]
        },
        'GERD': {
            "next_steps": [
                "Avoid trigger foods and drinks",
                "Eat smaller meals more frequently",
                "Consider over-the-counter antacids"
            ],
            "home_care": [
                "Elevate the head of your bed",
                "Avoid lying down after meals"
            ],
            "seek_medical_attention": [
                "Severe chest pain or difficulty swallowing",
                "Persistent symptoms despite treatment"
            ]
        },
        'Chronic cholestasis': {
            "next_steps": [
                "Consult a gastroenterologist",
                "Monitor liver function tests"
            ],
            "home_care": [
                "Maintain a healthy diet low in fats",
                "Stay hydrated"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain",
                "Jaundice or dark urine"
            ]
        },
        'Drug Reaction': {
            "next_steps": [
                "Stop taking the suspected drug immediately",
                "Consult a healthcare provider"
            ],
            "home_care": [
                "Keep the affected area clean",
                "Use antihistamines for mild reactions"
            ],
            "seek_medical_attention": [
                "Severe allergic reactions or anaphylaxis",
                "Worsening symptoms"
            ]
        },
        'Peptic ulcer disease': {
            "next_steps": [
                "Avoid NSAIDs and other irritants",
                "Consider proton pump inhibitors"
            ],
            "home_care": [
                "Eat a bland diet",
                "Avoid alcohol and smoking"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or vomiting blood",
                "Signs of perforation (sudden severe pain)"
            ]
        },
        'AIDS': {
            "next_steps": [
                "Consult an infectious disease specialist",
                "Start antiretroviral therapy (ART) as prescribed"
            ],
            "home_care": [
                "Maintain a healthy lifestyle and diet",
                "Regular monitoring of CD4 count"
            ],
            "seek_medical_attention": [
                "Signs of opportunistic infections",
                "Severe fatigue or weight loss"
            ]
        },
        'Diabetes': {
            "next_steps": [
                "Monitor blood sugar levels regularly",
                "Consult a dietitian for meal planning"
            ],
            "home_care": [
                "Exercise regularly and maintain a healthy weight",
                "Stay hydrated"
            ],
            "seek_medical_attention": [
                "Signs of hyperglycemia or hypoglycemia",
                "Uncontrolled blood sugar levels"
            ]
        },
        'Gastroenteritis': {
            "next_steps": [
                "Stay hydrated with oral rehydration solutions",
                "Monitor symptoms closely"
            ],
            "home_care": [
                "Eat bland foods as tolerated",
                "Avoid dairy and fatty foods"
            ],
            "seek_medical_attention": [
                "Severe dehydration or persistent vomiting",
                "Blood in stool"
            ]
        },
        'Bronchial Asthma': {
            "next_steps": [
                "Use rescue inhaler as needed",
                "Identify and avoid triggers"
            ],
            "home_care": [
                "Maintain an asthma action plan",
                "Use a humidifier if necessary"
            ],
            "seek_medical_attention": [
                "Severe shortness of breath",
                "Inability to speak full sentences"
            ]
        },
        'Hypertension': {
            "next_steps": [
                "Monitor blood pressure regularly",
                "Consult a healthcare provider for medication management"
            ],
            "home_care": [
                "Maintain a low-sodium diet",
                "Engage in regular physical activity"
            ],
            "seek_medical_attention": [
                "Severe headache or vision changes",
                "Chest pain or shortness of breath"
            ]
        },
        'Migraine': {
            "next_steps": [
                "Identify and avoid migraine triggers",
                "Consider over-the-counter pain relief"
            ],
            "home_care": [
                "Rest in a dark, quiet room",
                "Apply cold compresses to the forehead"
            ],
            "seek_medical_attention": [
                "Migraines that worsen or change in pattern",
                "Neurological symptoms like weakness or confusion"
            ]
        },
        'Cervical spondylosis': {
            "next_steps": [
                "Consult a physical therapist",
                "Consider pain management strategies"
            ],
            "home_care": [
                "Apply heat or cold packs to the neck",
                "Maintain good posture"
            ],
            "seek_medical_attention": [
                "Severe pain or neurological symptoms",
                "Loss of bladder or bowel control"
            ]
        },
        'Paralysis (brain hemorrhage)': {
            "next_steps": [
                "Seek immediate medical attention",
                "Follow up with a neurologist"
            ],
            "home_care": [
                "Supportive care as needed",
                "Rehabilitation therapy"
            ],
            "seek_medical_attention": [
                "Any signs of stroke (e.g., facial drooping, arm weakness)"
            ]
        },
        'Jaundice': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Monitor liver function tests"
            ],
            "home_care": [
                "Stay hydrated",
                "Avoid alcohol and fatty foods"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or confusion",
                "Rapid worsening of symptoms"
            ]
        },
        'Malaria': {
            "next_steps": [
                "Seek immediate medical attention",
                "Start antimalarial treatment as prescribed"
            ],
            "home_care": [
                "Stay hydrated and rest",
                "Monitor symptoms closely"
            ],
            "seek_medical_attention": [
                "Severe symptoms like high fever or confusion"
            ]
        },
        'Chicken pox': {
            "next_steps": [
                "Isolate to prevent spreading",
                "Consult a healthcare provider for antiviral treatment if needed"
            ],
            "home_care": [
                "Use calamine lotion for itching",
                "Stay hydrated"
            ],
            "seek_medical_attention": [
                "Severe symptoms or complications"
            ]
        },
        'Dengue': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Monitor for warning signs"
            ],
            "home_care": [
                "Stay hydrated with fluids",
                "Rest and avoid pain relievers like NSAIDs"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or persistent vomiting",
                "Bleeding or signs of shock"
            ]
        },
        'Typhoid': {
            "next_steps": [
                "Consult a healthcare provider for antibiotics",
                "Monitor symptoms closely"
            ],
            "home_care": [
                "Stay hydrated and rest",
                "Eat light, nutritious foods"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or persistent fever",
                "Signs of dehydration"
            ]
        },
        'Hepatitis A': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Monitor liver function tests"
            ],
            "home_care": [
                "Stay hydrated and rest",
                "Avoid alcohol and fatty foods"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or jaundice",
                "Rapid worsening of symptoms"
            ]
        },
        'Hepatitis B': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider antiviral therapy if indicated"
            ],
            "home_care": [
                "Maintain a healthy lifestyle and diet",
                "Regular monitoring of liver function"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or jaundice",
                "Signs of liver failure"
            ]
        },
        'Hepatitis C': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider antiviral therapy"
            ],
            "home_care": [
                "Maintain a healthy lifestyle and diet",
                "Regular monitoring of liver function"
            ],
            "seek_medical_attention": [
 "Severe abdominal pain or jaundice",
                "Signs of liver failure"
            ]
        },
        'Hepatitis D': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider antiviral therapy if indicated"
            ],
            "home_care": [
                "Maintain a healthy lifestyle and diet",
                "Regular monitoring of liver function"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or jaundice",
                "Signs of liver failure"
            ]
        },
        'Hepatitis E': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Monitor liver function tests"
            ],
            "home_care": [
                "Stay hydrated and rest",
                "Avoid alcohol and fatty foods"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or jaundice",
                "Rapid worsening of symptoms"
            ]
        },
        'Alcoholic hepatitis': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider hospitalization if severe"
            ],
            "home_care": [
                "Avoid alcohol completely",
                "Maintain a healthy diet"
            ],
            "seek_medical_attention": [
                "Severe abdominal pain or confusion",
                "Signs of liver failure"
            ]
        },
        'Tuberculosis': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Start antitubercular therapy as prescribed"
            ],
            "home_care": [
                "Stay hydrated and rest",
                "Monitor symptoms closely"
            ],
            "seek_medical_attention": [
                "Severe symptoms like coughing up blood",
                "Signs of respiratory distress"
            ]
        },
        'Common Cold': {
            "next_steps": [
                "Rest and stay hydrated",
                "Consider over-the-counter cold medications"
            ],
            "home_care": [
                "Use saline nasal sprays",
                "Gargle with warm salt water for sore throat"
            ],
            "seek_medical_attention": [
                "Symptoms lasting more than 10 days",
                "Difficulty breathing or chest pain"
            ]
        },
        'Pneumonia': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Start antibiotics if bacterial pneumonia is suspected"
            ],
            "home_care": [
                "Stay hydrated and rest",
                "Use a humidifier to ease breathing"
            ],
            "seek_medical_attention": [
                "Severe shortness of breath",
                "Chest pain or confusion"
            ]
        },
        'Dimorphic hemorrhoids (piles)': {
            "next_steps": [
                "Increase fiber intake",
                "Consult a healthcare provider for evaluation"
            ],
            "home_care": [
                "Use over-the-counter creams for relief",
                "Stay hydrated"
            ],
            "seek_medical_attention": [
                "Severe pain or bleeding",
                "Signs of infection"
            ]
        },
        'Heart attack': {
            "next_steps": [
                "Call emergency services immediately",
                "Chew aspirin if not allergic"
            ],
            "home_care": [
                "Stay calm and rest",
                "Avoid exertion"
            ],
            "seek_medical_attention": [
                "Any signs of chest pain or discomfort",
                "Shortness of breath or sweating"
            ]
        },
        'Varicose veins': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider compression stockings"
            ],
            "home_care": [
                "Elevate legs when resting",
                "Engage in regular physical activity"
            ],
            "seek_medical_attention": [
                "Severe pain or swelling",
                "Signs of blood clots"
            ]
        },
        'Hypothyroidism': {
            "next_steps": [
                "Consult an endocrinologist for evaluation",
                "Consider thyroid hormone replacement therapy"
            ],
            "home_care": [
                "Maintain a healthy diet",
                "Monitor symptoms closely"
            ],
            "seek_medical_attention": [
                "Severe fatigue or weight gain",
                "Signs of myxedema coma"
            ]
        },
        'Hyperthyroidism': {
            "next_steps": [
                "Consult an endocrinologist for evaluation",
                "Consider antithyroid medications"
            ],
            "home_care": [
                "Maintain a healthy diet",
                "Monitor symptoms closely"
            ],
            "seek_medical_attention": [
                "Severe anxiety or palpitations",
                "Signs of thyroid storm"
            ]
        },
        'Hypoglycemia': {
            "next_steps": [
                "Consume fast-acting carbohydrates",
                "Monitor blood sugar levels"
            ],
            "home_care": [
                "Maintain a healthy diet",
                "Regularly check blood sugar levels"
            ],
            "seek_medical_attention": [
                "Severe symptoms like confusion or seizures",
                "Inability to manage blood sugar levels"
            ]
        },
        'Osteoarthritis': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider physical therapy"
            ],
            "home_care": [
                "Maintain a healthy weight",
                "Engage in regular physical activity"
            ],
            "seek_medical_attention": [
                "Severe joint pain or swelling",
                "Signs of joint infection"
            ]
        },
        'Arthritis': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider disease-modifying antirheumatic drugs (DMARDs)"
            ],
            "home_care": [
                "Maintain a healthy lifestyle and diet",
                "Regular monitoring of joint symptoms"
            ],
            "seek_medical_attention": [
                "Severe joint pain or swelling",
                "Signs of joint infection"
            ]
        },
        '(vertigo) Paroxysmal Positional Vertigo': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider vestibular rehabilitation therapy"
            ],
            "home_care": [
                "Avoid triggers like certain head movements",
                "Use assistive devices for balance if needed"
            ],
            "seek_medical_attention": [
                "Severe vertigo or vomiting",
                "Signs of inner ear infection"
            ]
        },
        'Acne': {
            "next_steps": [
                "Maintain good skin hygiene",
                "Consider topical or oral acne treatments"
            ],
            "home_care": [
                "Avoid picking or popping pimples",
                "Use non-comedogenic products"
            ],
            "seek_medical_attention": [
                "Severe acne or scarring",
                "Signs of skin infection"
            ]
        },
        'Urinary tract infection': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Start antibiotics if prescribed"
            ],
            "home_care": [
                "Stay hydrated and urinate when needed",
                "Avoid irritants like certain foods or soaps"
            ],
            "seek_medical_attention": [
                "Severe symptoms like flank pain or vomiting",
                "Signs of sepsis"
            ]
        },
        'Psoriasis': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Consider topical or systemic treatments"
            ],
            "home_care": [
                "Maintain good skin hygiene",
                "Avoid triggers like stress or certain foods"
            ],
            "seek_medical_attention": [
                "Severe symptoms or flare-ups",
                "Signs of skin infection"
            ]
        },
        'Impetigo': {
            "next_steps": [
                "Consult a healthcare provider for evaluation",
                "Start antibiotics if prescribed"
            ],
            "home_care": [
                "Keep the affected area clean and covered",
                "Avoid close contact with others"
            ],
            "seek_medical_attention": [
                "Severe symptoms or spread of infection",
                "Signs of systemic infection"
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
