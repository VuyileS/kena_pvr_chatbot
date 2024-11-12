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
            )
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
        """
        return pd.read_sql_query(query, conn)

    # Load data and filter
    data = load_data()
    # data = data[data['COUNTRY'] == 'South Africa']
    data['timestamp'] = pd.to_datetime(data['CREATED_AT'])
    data = data.sort_values('timestamp')

    # User selects date range
    col1, col2 = st.columns(2)
    with col1: 
        start_date = st.date_input("Select start date", min_value=data['timestamp'].min().date())
    with col2:
        end_date = st.date_input("Select end date", max_value=data['timestamp'].max().date())

    # Dropdown filters for Gender, Age Group, and Operating System
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_gender = st.selectbox("Select Gender", options=['All'] + data['GENDER'].unique().tolist())
    with col2:
        selected_age_group = st.selectbox("Select Age Group", options=['All'] + data['AGE_GROUP'].unique().tolist())
    with col3:
        selected_os = st.selectbox("Select Operating System", options=['All'] + data['OPERATING_SYSTEM'].unique().tolist())
    filtered_data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)]
    # Apply Filters button
    if st.button("Apply Filters"):
        # with st.expander("You Can Review The Patient Data Here"):
        #     st.dataframe(filtered_data.sort_values(by="CREATED_AT", ascending=False))

        # Filter data based on date range and dropdown selections
        # filtered_data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)]
        if selected_gender != 'All':
            filtered_data = filtered_data[filtered_data['GENDER'] == selected_gender]
        if selected_age_group != 'All':
            filtered_data = filtered_data[filtered_data['AGE_GROUP'] == selected_age_group]
        if selected_os != 'All':
            filtered_data = filtered_data[filtered_data['OPERATING_SYSTEM'] == selected_os]
        with st.expander("You Can Review The Patient Data Here"):
            st.dataframe(filtered_data.sort_values(by="CREATED_AT", ascending=False))

         # Display total number of unique users for the selected filters and date range
        total_unique_users = filtered_data['PATIENT_ID'].nunique()
        st.metric(label="Total Users for Selected Duration and Filters", value=total_unique_users)
        # Plotting
        if not filtered_data.empty:
            filtered_data['animation_date'] = filtered_data['timestamp'].dt.date
            unique_dates = sorted(filtered_data['animation_date'].unique())
            unique_cities = filtered_data['CITY'].unique()

            # Initialize map
            fig = go.Figure()
            frames = []

            # Create traces for each city but set visibility to False initially
            traces = {}
            for city in unique_cities:
                trace = go.Scattermapbox(
                    lon=filtered_data[filtered_data['CITY'] == city]['LONGITUDE'],
                    lat=filtered_data[filtered_data['CITY'] == city]['LATITUDE'],
                    mode='markers',
                    marker=dict(size=10),
                    name=city,
                    visible=False  # Initially hide all cities
                )
                fig.add_trace(trace)
                traces[city] = trace

            # Create frames and control visibility of traces for each date
            for date in unique_dates:
                frame_data = filtered_data[filtered_data['animation_date'] <= date]
                frame_traces = []
                for city in unique_cities:
                    trace = traces[city]
                    if city in frame_data['CITY'].values:
                        # If city has data for this frame, show it
                        frame_traces.append(trace)
                        trace.visible = True
                    else:
                        trace.visible = False  # Hide if city not present in this frame
                frames.append(go.Frame(data=frame_traces, name=str(date)))

            fig.frames = frames

            # Set up initial map layout with progress bar
            fig.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    center=dict(lat=-30.5595, lon=22.9375),  # Center on South Africa
                    zoom=4,
                ),
                updatemenus=[{
                    "type": "buttons",
                    "showactive": True,  # Display progress bar
                    "direction": "left",  # Arrange buttons horizontally
                    # "x": 0.5,  # Center the buttons horizontally
                    # "y": -0.2,  # Position the buttons below the graph
                    "xanchor": "center",
                    "yanchor": "top",
                    "buttons": [{
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 2000, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
                    }, {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]
                    }]
                }],
                title_text="Geolocation Spread Over Time",
                showlegend=True
            )

            fig.update_layout(
                legend_title_text="City",
                legend=dict(
                    itemsizing="constant",
                    title_font=dict(size=12),
                )
            )

            st.plotly_chart(fig)
    if st.button("Generate User Profile with GPT"):
        # Calculate the most common values in the filtered data
        most_common_gender = filtered_data['GENDER'].mode()[0]
        most_common_age_group = filtered_data['AGE_GROUP'].mode()[0]
        most_common_os = filtered_data['OPERATING_SYSTEM'].mode()[0]
        most_common_city = filtered_data['CITY'].mode()[0]

        # Display the values
        # print(f"Most Common Gender: {most_common_gender}")
        # print(f"Most Common Age Group: {most_common_age_group}")
        # print(f"Most Common Operating System: {most_common_os}")
        # print(f"Most Common City: {most_common_city}")


        prompt = f"""
        Using the following common attributes from the dataset, construct a typical user profile.

        - Gender: {most_common_gender}
        - Age Group: {most_common_age_group}
        - Operating System: {most_common_os}
        - City: {most_common_city}

        Describe this typical user, considering possible behaviors, preferences, and lifestyle characteristics that align with someone of this demographic profile. Include details about how they might interact with the a telemedicine app called Kena app, reasons they might use it, and any notable characteristics.
        """

        # Your OpenAI API key
        # openai.api_key = "your_openai_api_key"
        client = OpenAI()
        # Generate profile using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that creates user profiles based on data patterns."},
                {"role": "user", "content": prompt}
            ]
        )

        # Display GPT-4's response
        # st.write(response.choices[0].message.content)
        # Display GPT-4's response with enhanced formatting
        profile_description = response.choices[0].message.content

        # Applying a simpler inline style for readability and minimal HTML
        st.markdown(f"""

                <p><strong>👤 Gender:</strong> {most_common_gender}</p>
                <p><strong>🎂 Age Group:</strong> {most_common_age_group}</p>
                <p><strong>💻 Operating System:</strong> {most_common_os}</p>
                <p><strong>📍 City:</strong> {most_common_city}</p>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif;">
                <h2 style="color: #4e5c6e;">Typical User Profile</h2>
                <p>{profile_description}</p>
            </div>
        """, unsafe_allow_html=True)