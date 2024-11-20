import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from geopy.distance import geodesic

def clinic_geolocation_spread(snowflake_username, snowflake_password):
    # Function to establish Snowflake connection and load data
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
    CIC.CONSULTATION_ID,
    CIC.CONSULTATION_ICD10_CODE_ID,
    CIC.ICD10_CODE_ID,
    CIC.CREATED_BY_USER_ID,
    IC.CODE,
    IC.DESCRIPTION,
    CONCAT_WS('-',IC.CODE, IC.DESCRIPTION )     AS CONSULTATION_ICD10S
    FROM ANALYTICS.PROD.STG_CLINIC__CONSULTATION_ICD10_CODES        CIC
    LEFT JOIN ANALYTICS.PROD.STG_CLINIC__ICD10_CODES        IC
        ON CIC.ICD10_CODE_ID = IC.ICD10_CODE_ID
    )
            SELECT
            PVR.CONSULTATION_ID,
            PVR.CREATED_AT,
            PVR.PATIENT_ID,
            D.CONSULTATION_ICD10S,
            PVR.AGE_AT_CONSULT,
            PVR.AGE_CATEGORY,
            PVR.GENDER,
            CASE
                WHEN PVR.VISITS = 1 THEN 'Once'
                WHEN PVR.VISITS > 1 THEN 'More Than Once'
            END AS VISITS,
            PVR.CENTRE_NAME,
            PVR.DISCHEM_SITE_CODE,
            PVR.PRACTICE_NUMBER,
            PVR.PRACTICE_PROVINCE,
            DRM.CITY,
            DRM.LATITUDE,
            DRM.LONGITUDE,
            DRM.REGIONAL_MANAGER_NAME,
            CASE
                WHEN PVR.DOCTOR IS NULL THEN 'Nurse Only'
                ELSE 'Nurse and Doctor'
            END AS CONSULTATION_TYPE
        FROM ANALYTICS.PROD.DIM_CLINIC__PATIENT_VISIT_REPORT PVR
        JOIN RAW.STATIC_DATA.DISCHEM_REGION_MANAGERS DRM
            ON PVR.PRACTICE_NUMBER = DRM.PRACTICE_NUMBER
            AND DRM.STATUS = 'Active'
        LEFT JOIN DATA              D 
            ON PVR.CONSULTATION_ID = D.CONSULTATION_ID
        WHERE DOCTOR IS NOT NULL
        """
        return pd.read_sql_query(query, conn)

    # Load data and filter
    data = load_data()

    # Load the data
    st.title("Regional Consultation Analytics Dashboard")
    consultation_data = data

    # Add Filters for Animated Map in the Main Pane
    st.write("### Filters for Growth of Consultations Over Time")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", consultation_data['CREATED_AT'].min().date())
    with col2:
        end_date = st.date_input("End Date", consultation_data['CREATED_AT'].max().date())
    with col3:
        time_granularity = st.selectbox("Granularity", ['Daily', 'Weekly', 'Monthly', 'Yearly'])

    selected_icd10 = st.selectbox("Select Diagnosis", options=['All'] + consultation_data['CONSULTATION_ICD10S'].unique().tolist())
    # Filter data based on the selected date range
    filtered_data = consultation_data[(consultation_data['CREATED_AT'] >= pd.to_datetime(start_date)) &
                                       (consultation_data['CREATED_AT'] <= pd.to_datetime(end_date))]

    # Preprocessing for visualizations
    filtered_data['Created_Month'] = filtered_data['CREATED_AT'].dt.to_period('M').dt.to_timestamp()
    filtered_data['Created_Week'] = filtered_data['CREATED_AT'].dt.to_period('W').dt.to_timestamp()
    filtered_data['Created_Year'] = filtered_data['CREATED_AT'].dt.to_period('Y').dt.to_timestamp()

    # Define time frame column based on granularity
    if time_granularity == 'Daily':
        filtered_data['TimeFrame'] = filtered_data['CREATED_AT'].dt.date
    elif time_granularity == 'Weekly':
        filtered_data['TimeFrame'] = filtered_data['Created_Week']
    elif time_granularity == 'Monthly':
        filtered_data['TimeFrame'] = filtered_data['Created_Month']
    elif time_granularity == 'Yearly':
        filtered_data['TimeFrame'] = filtered_data['Created_Year']

    if selected_icd10 != 'All':
            filtered_data = filtered_data[filtered_data['CONSULTATION_ICD10S'] == selected_icd10]
    # Animated Bubble Map: Growth of Consultations Over Time
    st.write("### Growth of Consultations Over Time")
    time_based_data = filtered_data.groupby(['TimeFrame', 'CITY', 'CENTRE_NAME', 'REGIONAL_MANAGER_NAME', 'LATITUDE', 'LONGITUDE']).size().reset_index(name='Consultations')
    fig = px.scatter_mapbox(
        time_based_data, lat='LATITUDE', lon='LONGITUDE', size='Consultations',
        color='Consultations', animation_frame='TimeFrame', size_max=40,
        mapbox_style="carto-positron", zoom=5, center=dict(lat=-30.5595, lon=22.9375),
        title=f"Consultation Growth Over Time ({time_granularity})",
        hover_data={'CENTRE_NAME': True,'CITY': True, 'REGIONAL_MANAGER_NAME': True}
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Consultations"))
    st.plotly_chart(fig)


    # Calculate Distances
    st.write("### Distance Analysis Between Medical Centres")
    threshold = st.number_input("High Consultation Threshold", min_value=1, value=500)
    high_consultations = time_based_data[time_based_data['Consultations'] >= threshold]
    low_consultations = time_based_data[time_based_data['Consultations'] < threshold]

    def calculate_distances(df1, df2):
        distances = []
        for _, row1 in df1.iterrows():
            for _, row2 in df2.iterrows():
                if row1['CENTRE_NAME'] != row2['CENTRE_NAME']:
                    distance = geodesic((row1['LATITUDE'], row1['LONGITUDE']), 
                                        (row2['LATITUDE'], row2['LONGITUDE'])).kilometers
                    distances.append({
                        'From': row1['CENTRE_NAME'],
                        'To': row2['CENTRE_NAME'],
                        'Distance (km)': round(distance, 2)
                    })
        return pd.DataFrame(distances)

    distance_df = calculate_distances(high_consultations, low_consultations)
    st.write("#### Distances Between High and Low Consultation Centres")
    st.dataframe(distance_df)

    # Rural vs Urban Classification
    st.write("### Rural vs Urban Classification")
    urban_cities = ['Johannesburg', 'Cape Town', 'Durban', 'Pretoria', 'Port Elizabeth']  # Example urban cities
    time_based_data['Area_Type'] = time_based_data['CITY'].apply(lambda x: 'Urban' if x in urban_cities else 'Rural')

    area_summary = time_based_data.groupby('Area_Type')['Consultations'].sum().reset_index()
    fig = px.pie(area_summary, names='Area_Type', values='Consultations', title="Rural vs Urban Consultations")
    st.plotly_chart(fig)
    
