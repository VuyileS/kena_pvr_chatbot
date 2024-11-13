import streamlit as st
import pandas as pd
import os
import snowflake.connector
import plotly.graph_objs as go
from datetime import datetime
import openai
from openai import OpenAI

def geolocation_spread(snowflake_username, snowflake_password):
    st.title("Kena Registered Users Geolocation Spread Over Time")

    # Establish Snowflake connection
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
                    PATIENT_ID,
                    IP_ADDRESS,
                    CITY,
                    REGION,
                    COUNTRY,
                    COUNTRY_CODE,
                    TIMEZONE,
                    MOBILE,
                    LON AS LONGITUDE,
                    LAT AS LATITUDE,
                    STATUS
                FROM ANALYTICS.PROD.stg_static__kena_users_ip_location
            ),
    PAYMENTS    AS (
    SELECT
    I.PATIENT_ID,
    I.INVOICE_ID,
    PICD.CODE_DESCRIPTION_SHORT         AS PRIMARY_ICD10_CODE,
    I.CREATED_AT,
    KT.TRANSACTION_ID,
    DENSE_RANK() OVER(PARTITION BY I.PATIENT_ID ORDER BY I.CREATED_AT::DATE DESC) AS RN,
    DENSE_RANK() OVER(PARTITION BY KT.INVOICE_ID ORDER BY KT.TRANSACTION_PAID_AT::DATE DESC) AS KTRN,
    FROM ANALYTICS.PROD.STG_KENA__INVOICES      I
    JOIN ANALYTICS.PROD.STG_KENA__INVOICE_ITEMS II
        ON I.INVOICE_ID = II.INVOICE_ID
        AND II.CONSULTATION_TYPE <> 'no_charge'
    LEFT JOIN ANALYTICS.PROD.STG_KENA__TRANSACTIONS KT
        ON I.INVOICE_ID = KT.INVOICE_ID
        AND KT.STATUS = 'success'
        AND KT.TRANSACTION_PAID_AT IS NOT NULL
    LEFT JOIN ANALYTICS.PROD.STG_KENA__PRIMARY_ICD10_CODES      PICD
        ON I.CONSULTATION_ID = PICD.CONSULTATION_ID
        ORDER BY I.PATIENT_ID ASC, I.CREATED_AT ASC
)
,CONSOLIDATED         AS (
            SELECT
                U.USER_ID           AS PATIENT_ID,
                U.CREATED_AT AS CREATED_AT,
                U.OPERATING_SYSTEM,
                U.AGE_GROUP,
                CASE
                    WHEN U.SEX = 0 THEN 'Male'
                    WHEN U.SEX = 1 THEN 'Female'
                    ELSE 'Unspecified'
                END AS GENDER,
                D.CITY AS CITY,
                D.REGION AS REGION,
                D.COUNTRY,
                D.LONGITUDE AS LONGITUDE,
                D.LATITUDE AS LATITUDE
            FROM ANALYTICS.PROD.STG_KENA_CLINIC__USERS U
            JOIN DATA D
            ON U.USER_ID = D.PATIENT_ID
            WHERE U.CURRENT_STEP = 99 AND U.ACCOUNT_STATUS = 1
)
SELECT
    DISTINCT C.*,
    CASE
        WHEN P.PATIENT_ID IS NOT NULL THEN 'Yes'
        ELSE 'No'
    END AS DID_CONSULT,
    CASE
        WHEN P.TRANSACTION_ID IS NOT NULL THEN 'Yes'
        ELSE 'No'
    END AS PAYMENT_MADE,
    P.PRIMARY_ICD10_CODE        AS LAST_CONSULT_PRIMARY_ICD10_CODE
FROM CONSOLIDATED C
LEFT JOIN PAYMENTS P
    ON C.PATIENT_ID = P.PATIENT_ID
    AND P.RN = 1
    ORDER BY C.PATIENT_ID ASC
        """
        return pd.read_sql_query(query, conn)

    # Load data and filter
    data = load_data()
    data['timestamp'] = pd.to_datetime(data['CREATED_AT'])
    data = data.sort_values('timestamp')

    # Initialize session state for filters
    if 'filtered_data' not in st.session_state:
        st.session_state['filtered_data'] = None
    if 'filters_applied' not in st.session_state:
        st.session_state['filters_applied'] = False

    # Date range and filter inputs
    col1, col2 = st.columns(2)
    with col1: 
        start_date = st.date_input("Select start date", min_value=data['timestamp'].min().date())
    with col2:
        end_date = st.date_input("Select end date", max_value=data['timestamp'].max().date())
    
    col1, col2 = st.columns(2)
    with col1:
        did_consult = st.selectbox("Did User Consult", options=['All'] + data['DID_CONSULT'].unique().tolist())
    with col2:
        did_pay = st.selectbox("Did User Make Payment", options=['All'] + data['PAYMENT_MADE'].unique().tolist())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_gender = st.selectbox("Select Gender", options=['All'] + data['GENDER'].unique().tolist())
    with col2:
        selected_age_group = st.selectbox("Select Age Group", options=['All'] + data['AGE_GROUP'].unique().tolist())
    with col3:
        selected_os = st.selectbox("Select Operating System", options=['All'] + data['OPERATING_SYSTEM'].unique().tolist())
    
    selected_icd10 = st.selectbox("Select Last Consult ICD10 Code", options=['All'] + data['LAST_CONSULT_PRIMARY_ICD10_CODE'].unique().tolist())
    
    # Only process data when "Apply Filters" button is clicked
    if st.button("Apply Filters"):
        # Filter data based on selections
        filtered_data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)]
        if did_consult != 'All':
            filtered_data = filtered_data[filtered_data['DID_CONSULT'] == did_consult]
        if did_pay != 'All':
            filtered_data = filtered_data[filtered_data['PAYMENT_MADE'] == did_pay]
        if selected_gender != 'All':
            filtered_data = filtered_data[filtered_data['GENDER'] == selected_gender]
        if selected_age_group != 'All':
            filtered_data = filtered_data[filtered_data['AGE_GROUP'] == selected_age_group]
        if selected_os != 'All':
            filtered_data = filtered_data[filtered_data['OPERATING_SYSTEM'] == selected_os]
        if selected_icd10 != 'All':
            filtered_data = filtered_data[filtered_data['LAST_CONSULT_PRIMARY_ICD10_CODE'] == selected_icd10]
        
        # Store filtered data in session state for reuse
        st.session_state['filtered_data'] = filtered_data
        st.session_state['filters_applied'] = True

    # Display filtered data and map if filters are applied
    if st.session_state['filters_applied'] and st.session_state['filtered_data'] is not None:
        filtered_data = st.session_state['filtered_data']

        # Data preview
        with st.expander("You Can Review The Patient Data Here"):
            st.dataframe(filtered_data.sort_values(by="CREATED_AT", ascending=False))

        # Display total number of unique users
        total_unique_users = filtered_data['PATIENT_ID'].nunique()
        st.metric(label="Total Users for Selected Duration and Filters", value=total_unique_users)

        # Map plotting
        filtered_data['animation_date'] = filtered_data['timestamp'].dt.date
        unique_dates = sorted(filtered_data['animation_date'].unique())
        unique_cities = filtered_data['CITY'].unique()

        fig = go.Figure()
        frames = []
        traces = {}

        # Create traces for each city
        for city in unique_cities:
            trace = go.Scattermapbox(
                lon=filtered_data[filtered_data['CITY'] == city]['LONGITUDE'],
                lat=filtered_data[filtered_data['CITY'] == city]['LATITUDE'],
                mode='markers',
                marker=dict(size=10),
                name=city,
                visible=False
            )
            fig.add_trace(trace)
            traces[city] = trace

        # Create frames and control visibility
        for date in unique_dates:
            frame_data = filtered_data[filtered_data['animation_date'] <= date]
            cumulative_count = len(frame_data)
            frame_traces = []

            for city in unique_cities:
                trace = traces[city]
                trace.visible = city in frame_data['CITY'].values
                frame_traces.append(trace)

            # Add annotation for date and cumulative count
            frame = go.Frame(
                data=frame_traces,
                name=str(date),
                layout=go.Layout(
                    annotations=[
                        go.layout.Annotation(
                            x=0.5,
                            y=1.1,
                            xref="paper",
                            yref="paper",
                            text=f"Date: {date} | Cumulative Users: {cumulative_count}",
                            showarrow=False,
                            font=dict(size=16)
                        )
                    ]
                )
            )
            frames.append(frame)

        fig.frames = frames

        fig.update_layout(
            mapbox=dict(style="carto-positron", center=dict(lat=-30.5595, lon=22.9375), zoom=4),
            updatemenus=[{
                "type": "buttons",
                "showactive": True,
                "direction": "left",
                "x": 0.5,
                "y": -0.1,
                "xanchor": "center",
                "yanchor": "top",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 2000, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]}
                ]
            }],
            title_text="Geolocation Spread Over Time",
            showlegend=True
        )

        fig.update_layout(
            legend_title_text="City",
            legend=dict(itemsizing="constant", title_font=dict(size=12))
        )

        st.plotly_chart(fig)

    # Generate User Profile with GPT
    if st.session_state['filters_applied'] and st.session_state['filtered_data'] is not None:
        if st.button("Generate User Profile with GPT"):
            filtered_data = st.session_state['filtered_data']
            most_common_gender = filtered_data['GENDER'].mode()[0]
            most_common_age_group = filtered_data['AGE_GROUP'].mode()[0]
            most_common_os = filtered_data['OPERATING_SYSTEM'].mode()[0]
            most_common_city = filtered_data['CITY'].mode()[0]
            most_icd10_code = filtered_data['LAST_CONSULT_PRIMARY_ICD10_CODE'].mode()[0]
            start_date, end_date = filtered_data['timestamp'].dt.date.min(), filtered_data['timestamp'].dt.date.max()

            prompt = f"""
            Using the following common attributes from the dataset, construct a short (less than 13 lines) typical user profile.

            - Gender: {most_common_gender}
            - Age Group: {most_common_age_group}
            - Operating System: {most_common_os}
            - City: {most_common_city}
            - Most Common ICD10 Code: {most_icd10_code}
            - Data Date Range: {start_date} to {end_date}

            Describe this typical user, considering possible behaviors, preferences, and lifestyle characteristics that align with someone of this demographic profile. Include details about how they might interact with a telemedicine app called Kena app, reasons they might use it, and any notable characteristics. Based on the date range provided, identify any seasonal factors that may influence the use of the app or common health concerns during this period.
            """

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant that creates user profiles based on data patterns."}, {"role": "user", "content": prompt}]
            )

            profile_description = response.choices[0].message.content

            st.markdown(f"""
                    <p><strong>👤 Gender:</strong> {most_common_gender}</p>
                    <p><strong>🎂 Age Group:</strong> {most_common_age_group}</p>
                    <p><strong>💻 Operating System:</strong> {most_common_os}</p>
                    <p><strong>📍 City:</strong> {most_common_city}</p>
                    <p><strong>👨‍🔬 Most Common ICD10 Code:</strong> {most_icd10_code}</p>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif;">
                    <h2 style="color: #4e5c6e;">Typical User Profile</h2>
                    <p>{profile_description}</p>
                </div>
            """, unsafe_allow_html=True)
