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
        SELECT DISTINCT
            CC.CONVERSATION_ID,
            CC.CONSULTATION_ID,
            CC.PATIENT_ID,
            CC.CREATED_AT,
            DATEDIFF(YEAR, U.DATE_OF_BIRTH, CC.CREATED_AT)      AS AGE,
            CC.CATEGORY,
            LB.RESPONSES,
            CC.STAFF_NAME,
            CC.VIDEOMED_CALL_TYPE,
            CCN.DATA
        FROM ANALYTICS.PROD.DIM_KENA__CLIENT_CONVERSATIONS CC
        LEFT JOIN ANALYTICS.PROD.STG_KENA__CLIENT_CLINIC_LINDA_BOTS LB
            ON CC.CONVERSATION_ID = LB.CONVERSATION_ID
        LEFT JOIN RAW.CLINIC_PUBLIC.CONSULTATION_CLINICAL_NOTES CCN
            ON CC.CONSULTATION_ID = CCN.CONSULTATION_ID
            AND CC.STAFF_ID = CCN.USER_ID
        LEFT JOIN ANALYTICS.PROD.STG_KENA_CLINIC__USERS     U 
            ON CC.PATIENT_ID = U.USER_ID
        WHERE CC.VIDEOMED_CALL_TYPE IS NOT NULL
        ORDER BY CC.CREATED_AT DESC
        """
        return pd.read_sql_query(query, conn)

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
        patient_json = patient_data[['AGE', 'CATEGORY', 'RESPONSES', 'STAFF_NAME', 'VIDEOMED_CALL_TYPE', 'DATA']].to_dict(orient='records')

        # Display JSON in a collapsible section
        with st.expander("See Processed JSON for LLM"):
            st.json(patient_json)

        # Generate and display support message
        if st.button("Generate Support Message"):
            support_message = generate_support_message(patient_json)
            st.subheader("Support Message")
            st.write(support_message)
