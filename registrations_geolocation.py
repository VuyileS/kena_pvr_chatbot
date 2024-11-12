import streamlit as st
import pandas as pd
import os
import snowflake.connector
import plotly.express as px
from datetime import datetime
import openai
from openai import OpenAI

def geolocation_spread(snowflake_username, snowflake_password):
# Set up the Streamlit app
    st.title("Kena Registered Users Geolocation Spread Over Time")

    # Establish your Snowflake connection
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
                U.CREATED_AT::DATE AS CREATED_AT,
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
        return pd.read_sql_query(query, con=conn)

    # Load data
    data = load_data()

    # Filter for South Africa users
    data = data[data['COUNTRY'] == 'South Africa']

    # Convert to datetime and sort
    data['timestamp'] = pd.to_datetime(data['CREATED_AT'])
    data = data.sort_values('timestamp')

    # User selects date range
    col1, col2 = st.columns(2)

    with col1: 
        start_date = st.date_input("Select start date", min_value=data['timestamp'].min().date())
    with col2:
        end_date = st.date_input("Select end date", max_value=data['timestamp'].max().date())
    col1, col2, col3 = st.columns(3)
    # Dropdown filters for Gender, Age Group, and Operating System
    with col1:
        selected_gender = st.selectbox("Select Gender", options=['All'] + data['GENDER'].unique().tolist())
    with col2:
        selected_age_group = st.selectbox("Select Age Group", options=['All'] + data['AGE_GROUP'].unique().tolist())
    with col3:
        selected_os = st.selectbox("Select Operating System", options=['All'] + data['OPERATING_SYSTEM'].unique().tolist())

    # Filter data based on date range and dropdown selections
    filtered_data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)]

    if selected_gender != 'All':
        filtered_data = filtered_data[filtered_data['GENDER'] == selected_gender]

    if selected_age_group != 'All':
        filtered_data = filtered_data[filtered_data['AGE_GROUP'] == selected_age_group]

    if selected_os != 'All':
        filtered_data = filtered_data[filtered_data['OPERATING_SYSTEM'] == selected_os]

    # Display total number of unique users for the selected filters and date range
    total_unique_users = filtered_data['PATIENT_ID'].nunique()
    st.metric(label="Total Users for Selected Duration and Filters", value=total_unique_users)

    # Check if there is data in the filtered range
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # Calculate cumulative user count per day
        filtered_data['animation_date'] = filtered_data['timestamp'].dt.date
        cumulative_data = pd.DataFrame()

        user_counts = []
        for date in sorted(filtered_data['animation_date'].unique()):
            current_data = filtered_data[filtered_data['animation_date'] <= date].copy()
            current_data['animation_frame'] = date.strftime('%Y-%m-%d')  # Convert to string format for animation
            
            # Track cumulative user count
            cumulative_user_count = len(current_data)
            user_counts.append({'date': date, 'user_count': cumulative_user_count})
            
            # Set city labels only for the most recent point in each frame
            current_data['label'] = ""
            if not current_data.empty:
                # Set the label for the most recent point only
                current_data.loc[current_data.index[-1], 'label'] = current_data.loc[current_data.index[-1], 'CITY']
            
            cumulative_data = pd.concat([cumulative_data, current_data])

        # Convert user counts to DataFrame for easy lookup in annotations
        user_counts_df = pd.DataFrame(user_counts)

        # Plot with Plotly Express with cumulative animation frames
        fig = px.scatter_mapbox(
            cumulative_data,
            lon='LONGITUDE',
            lat='LATITUDE',
            text='label',  # Use the 'label' column to display city names temporarily
            hover_name='CITY',
            hover_data={'REGION': True, 'timestamp': True},
            color='CITY',
            animation_frame="animation_frame",
            mapbox_style="carto-positron",
            zoom=5,
            center={"lat": -30.5595, "lon": 22.9375}  # Center on South Africa
        )
        
        # Add dynamic title with date and user count
        fig.update_layout(
            title={
                'text': '.',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin={"r":0,"t":40,"l":0,"b":0},
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 2000, "redraw": True}, "fromcurrent": True}]
                }, {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]
                }]
            }]
        )

        # Update annotations to display total users and date for each frame
        frames = fig.frames
        for frame in frames:
            frame_date = frame.name  # The date for this frame in 'YYYY-MM-DD' format
            user_count = user_counts_df.loc[user_counts_df['date'] == pd.to_datetime(frame_date).date(), 'user_count'].values[0]
            frame.layout.update(
                annotations=[
                    dict(
                        x=0.5,
                        y=1.1,
                        xref='paper',
                        yref='paper',
                        showarrow=False,
                        text=f"Date: {frame_date} | Total Users: {user_count}",
                        font=dict(size=16, color="black")
                    )
                ]
            )

        # Display map plot with the animated title
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
