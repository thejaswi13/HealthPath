# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering

# Load and preprocess the synthetic health dataset
def load_and_preprocess_data(file_path='health_data_synthetic.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Dataset 'health_data_synthetic.csv' not found. Please ensure it exists.")
        return None, None, None

    categorical_cols = ['Chronic_Condition', 'Diet_Type', 'Smoking_Habit', 'Menstrual_Cycle_Regularity', 'Stress_Level', 'Tech_Engagement']
    for col in categorical_cols:
        df[col] = df[col].fillna('None')
        unique_values = df[col].unique().tolist()
        if 'None' not in unique_values:
            unique_values.append('None')
        df[col] = pd.Categorical(df[col], categories=unique_values)
   
    le = LabelEncoder()
    encoded_cats = {}
    for col in categorical_cols:
        encoded_cats[col] = le.fit(df[col])
        df[col] = le.transform(df[col])
   
    numerical_cols = ['Age', 'BMI', 'Physical_Activity_Hours_Per_Week', 'Mental_Health_Score', 'Sleep_Hours_Per_Night', 'Alcohol_Consumption_Per_Week']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
   
    return df, scaler, encoded_cats

# Analyze clusters and create a meaningful group mapping
def analyze_and_map_clusters(data_df, cluster_labels):
    data_df['Cluster'] = cluster_labels
    centroids = data_df.groupby('Cluster').mean()
   
    risk_scores = {}
    for cluster in centroids.index:
        risk = (
            centroids.loc[cluster, 'BMI'] +
            -centroids.loc[cluster, 'Sleep_Hours_Per_Night'] +
            centroids.loc[cluster, 'Stress_Level'] +
            -centroids.loc[cluster, 'Mental_Health_Score']
        )
        risk_scores[cluster] = risk
   
    sorted_clusters = sorted(risk_scores.items(), key=lambda x: x[1])
    group_mapping = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}
   
    return group_mapping

# Predict user's group
def predict_user_group(user_data, data_df, scaler, le_dict, hierarchical_model, group_mapping):
    user_df = pd.DataFrame([user_data])
   
    categorical_cols = ['Chronic_Condition', 'Diet_Type', 'Smoking_Habit', 'Menstrual_Cycle_Regularity', 'Stress_Level', 'Tech_Engagement']
    for col in categorical_cols:
        if col in user_data:
            value = user_data[col]
            if value == 'None' or pd.isna(value) or value not in le_dict[col].classes_:
                user_df[col] = 0
            else:
                user_df[col] = le_dict[col].transform([value])[0]
   
    numerical_cols = ['Age', 'BMI', 'Physical_Activity_Hours_Per_Week', 'Mental_Health_Score', 'Sleep_Hours_Per_Night', 'Alcohol_Consumption_Per_Week']
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])
   
    user_df = user_df[data_df.columns.drop('Cluster', errors='ignore')]
    combined_df = pd.concat([data_df.drop('Cluster', axis=1), user_df])
    cluster_labels = hierarchical_model.fit_predict(combined_df)
    user_cluster = cluster_labels[-1]
   
    return group_mapping[user_cluster]

# Page setup
st.set_page_config(page_title="HealthPath", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50; text-shadow: 2px 2px 4px #ccc;'>HealthPath – Your Unique Journey</h1>
    <p style='text-align: center; color: #666; font-style: italic;'>Real-time insights with your data-driven health assistant—clustering for personalized care!</p>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = None
if 'user_group' not in st.session_state:
    st.session_state['user_group'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Load and preprocess dataset
data_df, scaler, le_dict = load_and_preprocess_data()
if data_df is None:
    st.stop()

# Fit clustering model and create group mapping
hierarchical_model = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = hierarchical_model.fit_predict(data_df)
group_mapping = analyze_and_map_clusters(data_df, cluster_labels)

# User Input Form
st.markdown("<h3 style='text-align: center; color: #333;'>Tell Us About You</h3>", unsafe_allow_html=True)
with st.form(key='user_input', clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Thejaswi", help="What’s your name?")
        age = st.number_input("Age", value=21.0, step=0.1, help="How old are you?")
        bmi = st.number_input("BMI", value=27.5, step=0.1, help="Your Body Mass Index (weight in kg / height in m²)")
        sleep_hours = st.number_input("Sleep Hours", value=6.0, step=0.1, help="Average hours you sleep per night")
        chronic_condition = st.selectbox("Chronic Condition", ['None', 'Diabetes', 'Heart Disease', 'Hypertension', 'Other'],
                                        help="Select any chronic condition you have")
    with col2:
        physical_activity_hours = st.number_input("Physical Activity Hours/Week", value=5.0, step=0.1,
                                                help="Average hours of physical activity per week")
        stress_level = st.selectbox("Stress Level", ['Low', 'Medium', 'High'], help="How stressed do you feel daily?")
        mental_health_score = st.number_input("Mental Health Score (1-10)", value=2.0, min_value=1.0, max_value=10.0,
                                            help="Rate your mental health on a scale of 1-10")
        diet_type = st.selectbox("Diet Type", ['Balanced', 'Vegan', 'Vegetarian', 'High-Protein', 'Fast-Food'],
                                help="Your typical diet")
   
    submit_button = st.form_submit_button(label="Get My Health Insights")

# Clustering-Based Grouping and Recommendations
if submit_button:
    user_data = {
        'Age': age,
        'BMI': bmi,
        'Physical_Activity_Hours_Per_Week': physical_activity_hours,
        'Chronic_Condition': chronic_condition,
        'Mental_Health_Score': mental_health_score,
        'Sleep_Hours_Per_Night': sleep_hours,
        'Diet_Type': diet_type,
        'Smoking_Habit': 'Non-Smoker',
        'Alcohol_Consumption_Per_Week': 0,
        'Menstrual_Cycle_Regularity': 'Regular',
        'Stress_Level': stress_level,
        'Tech_Engagement': 'Medium'
    }

    with st.spinner("Analyzing your health data..."):
        her_group = predict_user_group(user_data, data_df, scaler, le_dict, hierarchical_model, group_mapping)

    st.session_state['user_profile'] = {
        'name': name, 'age': age, 'bmi': bmi, 'sleep_hours': sleep_hours, 'chronic_conditions': chronic_condition,
        'physical_activity_hours': physical_activity_hours, 'stress_level': stress_level,
        'mental_health_score': mental_health_score, 'diet_type': diet_type, 'group': her_group
    }
    st.session_state['user_group'] = her_group

    warnings = []
    if age < 0 or age > 120: warnings.append("Age seems off (0–120 is typical). Still processing!")
    if bmi < 0 or bmi > 60: warnings.append("BMI looks unusual (0–60 is typical). Proceeding anyway!")
    if sleep_hours < 0 or sleep_hours > 24: warnings.append("Sleep hours seem odd (0–24 is typical). Moving forward!")
    if physical_activity_hours < 0: warnings.append("Physical activity hours can’t be negative. Still processing!")
    if mental_health_score < 1 or mental_health_score > 10: warnings.append("Mental health score should be 1–10. Proceeding anyway!")
   
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("Looks good—here’s your report!")

    st.markdown("<h3 style='text-align: center; color: #333;'>Your Health Journey</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #555; text-align: center;'>
    <strong>What’s Your "Group"?</strong>  
    We use advanced clustering to sort you into groups from healthiest (Group 0) to needing more support (Group 3):  
    </p>
    <ul style='color: #555;'>
        <li><strong>Group 0</strong>: Top shape—minimal health worries!</li>
        <li><strong>Group 1</strong>: Pretty good, with minor tweaks needed.</li>
        <li><strong>Group 2</strong>: Managing challenges—room to improve!</li>
        <li><strong>Group 3</strong>: Bigger hurdles—we’ve got your back!</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown(f"<h4 style='color: #4CAF50; text-align: center;'>Hello, {name}!</h4>", unsafe_allow_html=True)
    st.write(f"You’re in **Group {her_group}**. Here’s your personalized health report:")

    if her_group == 0:
        st.success("**Health Insight:** You’re in awesome health! Keep up the great work!")
    elif her_group == 1:
        st.info("**Health Insight:** You’re doing well, but a few areas could use attention.")
    elif her_group == 2:
        st.warning("**Health Insight:** You’re facing some health challenges—let’s address them.")
    elif her_group == 3:
        st.error("**Health Insight:** You’ve got significant challenges—here’s how to tackle them.")

    st.markdown("**Your Action Plan:**", unsafe_allow_html=True)
    tips = []
    if her_group == 0:
        tips.append("Maintain your balanced lifestyle with regular check-ups and light exercise.")
    elif her_group == 1:
        if bmi > 25: tips.append(f"Your BMI of {bmi:.1f} is slightly high—try 15–20 minutes of brisk walking 3–4 times a week.")
        if sleep_hours < 7: tips.append(f"You’re getting {sleep_hours:.1f} hours of sleep—aim for 7–8 with a calm bedtime routine.")
        if stress_level in ['Medium', 'High']: tips.append(f"With {stress_level} stress, consider 5–10 minutes of daily deep breathing.")
    elif her_group == 2:
        if bmi >= 25: tips.append(f"Your BMI of {bmi:.1f} suggests you’re overweight—start with 20–30 minutes of daily walking and more veggies.")
        if sleep_hours < 7: tips.append(f"Only {sleep_hours:.1f} hours of sleep—target 7–8 hours by cutting screen time before bed.")
        if stress_level in ['Medium', 'High']: tips.append(f"{stress_level} stress? Try 10 minutes of meditation daily.")
        if mental_health_score <= 3: tips.append(f"Your mental health score of {mental_health_score} is low—consider talking to a friend or professional.")
    elif her_group == 3:
        if bmi > 30: tips.append(f"With a BMI of {bmi:.1f}, consult a nutritionist and aim for 20–30 minutes of daily activity.")
        if sleep_hours < 6: tips.append(f"Critical sleep of {sleep_hours:.1f} hours—prioritize 7–8 hours with a strict schedule.")
        if stress_level == 'High' or mental_health_score <= 5: tips.append("High stress or low mental health? Therapy could really help.")
        if chronic_condition != 'None': tips.append(f"For {chronic_condition}, regular doctor visits are key.")

    if not tips:
        tips.append("Let’s fine-tune your plan—everything looks borderline, so keep monitoring your habits!")
   
    for tip in tips:
        st.markdown(f"- {tip}")

# Chatbot section with static responses
st.markdown("<h3 style='text-align: center; color: #333;'>Chat with Your Health Assistant</h3>", unsafe_allow_html=True)
st.info("This is a Rule based chatbot")

user_query = st.text_input("Ask Anything!", "", help="What’s on your mind about your health?")
if user_query:
    if st.session_state['user_profile'] is not None:
        profile = st.session_state['user_profile']
        group = st.session_state['user_group']
        profile_str = (
            f"User: {profile['name']}, Age: {profile['age']}, BMI: {profile['bmi']}, "
            f"Sleep: {profile['sleep_hours']} hours, Condition: {profile['chronic_conditions']}, "
            f"Exercise: {profile['physical_activity_hours']} hours/week, Stress: {profile['stress_level']}, "
            f"Mental Health Score: {profile['mental_health_score']}/10, Diet: {profile['diet_type']}, "
            f"Group: {group}"
        )
        # Static response logic based on user query
        query_lower = user_query.lower()
        if "sleep" in query_lower:
            response = f"Hi {profile['name']}! With {profile['sleep_hours']} hours of sleep, aim for 7-8 hours nightly. Try a consistent bedtime routine."
        elif "stress" in query_lower:
            response = f"Hi {profile['name']}! For your {profile['stress_level']} stress, consider 10 minutes of meditation or deep breathing daily."
        elif "bmi" in query_lower or "weight" in query_lower:
            response = f"Hi {profile['name']}! Your BMI is {profile['bmi']:.1f}. {'Maintain it with regular exercise!' if profile['bmi'] < 25 else 'Try 20-30 minutes of daily walking to manage it.'}"
        elif "diet" in query_lower:
            response = f"Hi {profile['name']}! Your {profile['diet_type']} diet is great—{'keep it balanced!' if profile['diet_type'] == 'Balanced' else 'ensure you get enough nutrients!'}"
        elif "exercise" in query_lower or "activity" in query_lower:
            response = f"Hi {profile['name']}! You’re doing {profile['physical_activity_hours']} hours/week—{'awesome, keep it up!' if profile['physical_activity_hours'] >= 5 else 'aim for 5+ hours!'}"
        else:
            response = f"Hi {profile['name']}! Based on your profile (Group {group}), focus on maintaining your {profile['diet_type']} diet and {profile['physical_activity_hours']} hours of exercise!"
    else:
        response = "Hi there! Please fill out the form above so I can give you personalized health advice!"
    
    st.session_state['chat_history'].append(("You", user_query))
    st.session_state['chat_history'].append(("Health Assistant", response))

# Display Chat History
if st.session_state['chat_history']:
    st.markdown("**Chat History:**", unsafe_allow_html=True)
    for sender, message in st.session_state['chat_history']:
        if sender == "You":
            st.markdown(f"<p style='color: #4CAF50;'><strong>{sender}:</strong> {message}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color: #333;'><strong>{sender}:</strong> {message}</p>", unsafe_allow_html=True)

# Enhanced About the App Section
with st.expander("Discover HealthPath", expanded=False):
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px;'>
        <h3 style='color: #4CAF50; text-align: center;'>Welcome to Your Health Companion!</h3>
        <p style='color: #333; text-align: center;'>
            HealthPath is here to guide you on a personalized wellness journey, tailored just for you—especially for women like us!
        </p>
        <ul style='color: #555; list-style-type: none; padding-left: 0;'>
            <li>✨ <strong>Smart Insights:</strong> Our cutting-edge clustering tech analyzes your unique health data to deliver real-time, actionable advice.</li>
            <li>💬 <strong>Your AI Buddy:</strong> Chat with our friendly health assistant anytime—get tips, answers, and support that fit your life.</li>
            <li>💪 <strong>Empowerment Made Simple:</strong> No costs, no fuss—just practical steps to feel your best, from diet to stress and beyond.</li>
        </ul>
        <p style='color: #666; text-align: center; font-style: italic;'>
            Built with care for a healthier tomorrow—because your health matters!
        </p>
    </div>
    """, unsafe_allow_html=True)
