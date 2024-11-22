import streamlit as st
import pandas as pd
import snowflake.connector
import openai
import os
from datetime import datetime
from openai import OpenAI

# Snowflake connection setup
def patient_messaging(snowflake_username, snowflake_password ):
    conn = snowflake.connector.connect(
            user=snowflake_username,
            password=snowflake_password,
            account=os.getenv('snowflake_account'),
            warehouse=os.getenv('snowflake_warehouse'),
            database=os.getenv('snowflake_database'),
            schema=os.getenv('snowflake_schema')
        )

    # Query data from Snowflake
    @st.cache_data
    def load_data():
        query = """
        WITH DATA AS (
        SELECT
            CONVERSATION_ID,
            CONSULTATION_ID,
            PATIENT_ID,
            CREATED_AT,
            CATEGORY,
            STAFF_NAME, 
            STAFF_ID,
            VIDEOMED_CALL_TYPE,
            DENSE_RANK() OVER(PARTITION BY PATIENT_ID ORDER BY CREATED_AT DESC) AS NUMBER_CONSULTS
            FROM ANALYTICS.PROD.DIM_KENA__CLIENT_CONVERSATIONS
        )
        SELECT DISTINCT
            CC.CONVERSATION_ID,
            CC.CONSULTATION_ID,
            CC.PATIENT_ID,
            CC.CREATED_AT,
            DATEDIFF(YEAR, U.DATE_OF_BIRTH, CC.CREATED_AT)      AS AGE,
            CC.CATEGORY,
            LB.RESPONSES,
            CC.STAFF_NAME,
            CONCAT_WS(' ', U.FIRST_NAME, U.LAST_NAME)           AS PATIENT_NAME,
            CC.VIDEOMED_CALL_TYPE,
            CCN.DATA
        FROM DATA CC
        LEFT JOIN ANALYTICS.PROD.STG_KENA__CLIENT_CLINIC_LINDA_BOTS LB
            ON CC.CONVERSATION_ID = LB.CONVERSATION_ID
        LEFT JOIN RAW.CLINIC_PUBLIC.CONSULTATION_CLINICAL_NOTES CCN
            ON CC.CONSULTATION_ID = CCN.CONSULTATION_ID
            AND CC.STAFF_ID = CCN.USER_ID
        LEFT JOIN ANALYTICS.PROD.STG_KENA_CLINIC__USERS     U 
            ON CC.PATIENT_ID = U.USER_ID
        WHERE CC.VIDEOMED_CALL_TYPE IS NOT NULL AND CC.NUMBER_CONSULTS = 1
        ORDER BY CC.CREATED_AT DESC
        """
        return pd.read_sql_query(query, conn)
    @st.cache_data
    def load_ratings_data():
        query = """
        SELECT
            PATIENT_ID,
            CONVERSATION_ID,
            CREATED_AT,
            RATING,
            COMMENT
        FROM RAW.CLINIC_PUBLIC.CLIENT_CLINIC_CONVERSATION_RATINGS
        ORDER BY PATIENT_ID ASC, CREATED_AT ASC
        """
        return pd.read_sql_query(query, conn)

    # Load data
    # consultation_data = load_data()
    ratings_data = load_ratings_data()
    # Function to generate a support message from GPT
    def generate_support_message(patient_json):
        client = OpenAI()
        prompt = f"As a friendly and supportive telemedicine assistant with a goal to ensure the patient returns, highlight their consultation and give words of encouragement as a push notification or text message based on the following data: {patient_json}. Please try and keep it to around 10 lines. Include emojis as and when needed depending on their age. Also take their age in your response. The responses field indicates the responses the patient gave to the bot just before engaging with the clinicians. The DATA:note is the clinical note written by the clinician. Try to keep the length of your response in such a way it will be an easy read as a push notification."
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly and supportive telemedicine assistant working for Kena, with a goal to ensure the patient returns, highlight their consultation and give words of encouragement as a push notification or text message. Consider the data given as part of one consultation but in Kena patients have the ability to consult with multiple clinicians based on whether they need a nurse only or a doctor as well or a mental health practitioner or a clinical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    # Streamlit App
    # st.set_page_config(page_title="Patient Support Message Generator", page_icon="🩺")

    st.title("Patient Support Message Generator")
    st.write("Select a date range and patient to generate a personalized message from the Kena LLM Bot.")

    # Load data
    data = load_data()

    # Date range selection in the main pane, side-by-side columns
    col1, col2 = st.columns(2)
    with col1:
        min_date = data['CREATED_AT'].min().date()
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=data['CREATED_AT'].max().date())
    with col2:
        max_date = data['CREATED_AT'].max().date()
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Filter data by date range
    filtered_data = data[(data['CREATED_AT'] >= pd.to_datetime(start_date)) & (data['CREATED_AT'] <= pd.to_datetime(end_date))]

    # Display raw data in an expander
    with st.expander("You Can Review The Consultation Data Here"):
        st.dataframe(filtered_data.sort_values(by="CREATED_AT", ascending=False))

    # Patient selection based on filtered data
    patient_ids = filtered_data['PATIENT_ID'].unique().tolist()
    selected_patient_id = st.selectbox("Select a Patient ID", options=patient_ids)

    # Process selected patient data
    if selected_patient_id:
        patient_data = filtered_data[filtered_data['PATIENT_ID'] == selected_patient_id]
        # Prepare JSON with fields already in JSON format
        patient_json = patient_data[['PATIENT_NAME', 'AGE', 'CATEGORY', 'RESPONSES', 'STAFF_NAME', 'VIDEOMED_CALL_TYPE', 'DATA']].to_dict(orient='records')

        # Display JSON in a collapsible section
        with st.expander("See Processed JSON for LLM"):
            st.json(patient_json)

        # Generate and display support message
        if st.button("Generate Support Message"):
            support_message = generate_support_message(patient_json)
            st.subheader("Support Message")
            st.write(support_message)

        # st.title("Consultation History Overview")
        # st.write("Explore consultation history for patients with 2 or more consultations.")

        # # Filter ratings data to include only patients with 2 or more consultations
        # patient_consult_counts = ratings_data['PATIENT_ID'].value_counts()
        # eligible_patients = patient_consult_counts[patient_consult_counts >= 2].index
        # filtered_ratings_data = ratings_data[ratings_data['PATIENT_ID'].isin(eligible_patients)]

        # # Patient selection
        # overview_patient_ids = filtered_ratings_data['PATIENT_ID'].unique().tolist()
        # selected_overview_patient_id = st.selectbox("Select a Patient ID for History Overview", options=overview_patient_ids)

        # if selected_overview_patient_id:
        #     # Filter data for the selected patient
        #     patient_history = filtered_ratings_data[filtered_ratings_data['PATIENT_ID'] == selected_overview_patient_id]
        #     avg_rating = patient_history['RATING'].mean()
        #     combined_comments = " ".join(patient_history['COMMENT'].dropna().tolist())
        #     num_consultations = len(patient_history)
        #     consultation_dates = patient_history['CREATED_AT'].sort_values()
        #     time_between_consults = consultation_dates.diff().dt.days.dropna().tolist()
        #     days_since_last_consult = (pd.Timestamp.now() - consultation_dates.max()).days

        #     # Create Summary JSON
        #     history_summary = {
        #         "patient_id": selected_overview_patient_id,
        #         "average_rating": round(avg_rating, 2),
        #         # "combined_comments": combined_comments,
        #         "number_of_consultations": num_consultations,
        #         "time_between_consultations": time_between_consults,
        #         "days_since_last_consultation": days_since_last_consult,
        #         "ratings": patient_history[['RATING', 'COMMENT', 'CREATED_AT']].to_dict(orient="records")
        #     }

        #     # Display JSON Summary
        #     with st.expander("See Patient History Summary JSON"):
        #         st.json(history_summary)

        #     # Generate LLM Overview
        #     if st.button("Generate History Overview"):
        #         client = OpenAI()
        #         prompt = f"Using the following data: {history_summary}. Please summarize the user's average rating and craft a welcoming message reflecting on their consultation history. Highlight what they enjoyed about specific consultations on certain dates and acknowledge any negative experiences with empathy, ensuring the tone is warm and inviting. Mention how long it has been since their last visit to create a sense of connection. Keep the response concise (under 10 lines or one paragraph), engaging, and direct. Use emojis sparingly to add warmth without being overly extravagant."
        #         openai.api_key = os.getenv("OPENAI_API_KEY")

        #         response = client.chat.completions.create(
        #             model="gpt-4",
        #             messages=[
        #                 {"role": "system", "content": "You are a professional assistant to the Kena Health app welcoming patients. Don't mention words like average rating but do refer to it in an engaging manner. Also the patients do not have scheduled consultations, however we want to ensure that they feel a personal touch of their historic consultations. Your tone and words should lead the patient to actually consulting again. Do not include a sign off on the message "},
        #                 {"role": "user", "content": prompt}
        #             ]
        #         )
        #         overview_message = response.choices[0].message.content
        #         st.subheader("Consultation History Overview")
        #         st.write(overview_message)
