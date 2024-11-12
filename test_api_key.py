import requests
import pandas as pd
import logging
import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_ip_addresses_from_snowflake():
    '''
    This function connects to Snowflake, retrieves a list of patient IDs
    and their associated IP addresses from a specified table, and returns 
    them as a list of tuples.
    
    Returns
    -------
    list
        A list of tuples containing patient_id and their IP addresses.
    '''

    # Establish a connection to Snowflake
    conn = snowflake.connector.connect(
        user= os.getenv('snowflake_username'),
        password=os.getenv('snowflake_password'),
        account=os.getenv('snowflake_account'),
        warehouse=os.getenv('snowflake_warehouse'),
        database=os.getenv('snowflake_database'),
        schema=os.getenv('snowflake_schema')
    )
    try:
        # Create a cursor object
        cur = conn.cursor()

        # Write your SQL query to retrieve patient_id and IP addresses from your table
        query = """
            SELECT USER_ID  AS PATIENT_ID, CURRENT_SIGN_IN_IP 
            FROM ANALYTICS.PROD.STG_KENA_CLINIC__USERS 
            WHERE CURRENT_SIGN_IN_IP IS NOT NULL 
            AND CREATED_AT > '2024-10-18 11:52:23.271454000'
        """

        # Execute the query
        cur.execute(query)

        # Fetch all the patient IDs and IP addresses
        ip_addresses = [(row[0], row[1]) for row in cur.fetchall()]

        # Close the cursor and connection
        cur.close()

    finally:
        conn.close()

    return ip_addresses

def convert_ip_to_location(ip_list=[], params=[], api_key=""):
    '''
    This function takes a list of patient IDs and IP addresses, sends the IPs to
    an API service, and records the response which is associated with the location 
    of the given IP address. A pd.DataFrame will be returned with the patient IDs,
    their associated IP addresses, and the location parameters.
    
    Parameters
    ----------
    ip_list: list[tuple]
        A list of tuples containing patient_id and IP addresses.
    params: list[str]
        A list of the parameters we would like to receive back from
        the API when we make our request.
    api_key: str
        The API key for the paid subscription of ip-api.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame that contains the patient ID, IP address, and 
        location information retrieved for each from the API.
    '''
    
    valid_params = ['status', 'message', 'continent', 'continentCode', 'country',
                    'countryCode', 'region', 'regionName', 'city', 'district', 
                    'zip', 'lat', 'lon', 'timezone', 'offset', 'currency', 'isp',
                    'org', 'as', 'asname', 'reverse', 'mobile', 'proxy', 'hosting',
                    'query']

    # input checks
    assert isinstance(ip_list, list), 'The ip_list must be passed in a list of tuples'
    assert ip_list, 'You must pass at least one IP address to the function'
    assert isinstance(params, list), 'You must pass at least one parameter'
    for param in params:
        assert param in valid_params, f"{param} is not a valid parameter. List of valid params: {valid_params}"

    # the base URL for the API to connect to (JSON response)
    url = 'http://pro.ip-api.com/json/'

    # specify query parameters and convert to properly formatted search string
    params_string = ','.join(params)

    # create a dataframe to store the responses
    df = pd.DataFrame(columns=['patient_id', 'ip_address'] + params)

    # Process each patient_id and IP in the list
    for patient_id, ip in ip_list:
        resp = requests.get(url + ip, params={'fields': params_string, 'key': api_key})
        info = resp.json()
        if info["status"] == 'success':
            # if response is okay, create a DataFrame row and concat
            info.update({'patient_id': patient_id, 'ip_address': ip})
            df_row = pd.DataFrame([info])
            df = pd.concat([df, df_row], ignore_index=True)
        else:
            # if there was a problem with the response, trigger a warning
            logging.warning(f'Unsuccessful response for IP: {ip} (Patient ID: {patient_id})')
    
    # return the dataframe with all the information
    return df

# Main script to fetch IPs and patient IDs from Snowflake and convert to location data

# Fetch IP addresses and patient IDs from Snowflake table
ip_list = fetch_ip_addresses_from_snowflake()

# Set your API key for the paid IP-API service
# api_key = os.getenv('ip_api_key')  # Alternatively, hardcode your API key here
api_key = 'YzCYTTWn5Pq8Xcb'
print(f"Using API key: {api_key}")

# Now pass the patient_id and IP list to the conversion function with the API key
df = convert_ip_to_location(
    ip_list=ip_list,
    params=['city', 'region', 'country', 'countryCode', 'timezone', 'mobile', 'lat', 'lon', 'status'],
    api_key=api_key  # Pass your API key here
)

# Print the result
print(df.head())

# Write the DataFrame to a CSV file
df.to_csv('ip_location_data_with_patient_ids.csv', index=False)
