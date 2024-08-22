import streamlit as st
import openai
from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine, inspect

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from snowflake.snowpark import Session

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app title
st.title("Kena PVR Chatbot with SQL Integration")

# Initialize OpenAI client
client = openai

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Kena PVR chatbot. How can I assist you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize Snowflake connection parameters
    connection_parameters = {
        "account": os.getenv('snowflake_account'),
        "user": os.getenv('snowflake_username'),
        "password": os.getenv('snowflake_password'),
        "warehouse": os.getenv('snowflake_warehouse'),
        "database": os.getenv('snowflake_database'),
        "schema": os.getenv('snowflake_schema')
    }

    snowflake_url = f"snowflake://{os.getenv('snowflake_username')}:{os.getenv('snowflake_password')}@{os.getenv('snowflake_account')}/{os.getenv('snowflake_database')}/{os.getenv('snowflake_schema')}?warehouse={os.getenv('snowflake_warehouse')}"

    # Create SQLAlchemy engine and inspect the database
    engine = create_engine(snowflake_url)
    inspector = inspect(engine)

    # List all tables in the specified schema to verify the table existence
    tables = inspector.get_table_names(schema=os.getenv('snowflake_schema'))

    if 'dim_kena__patient_visit_report' not in tables:
        raise ValueError(f"Table 'dim_kena__patient_visit_report' not found in schema '{os.getenv('snowflake_schema')}'.")

    # Initialize SQLDatabase from LangChain
    db = SQLDatabase.from_uri(snowflake_url, sample_rows_in_table_info=1, include_tables=['dim_kena__patient_visit_report'])

    # Initialize the OpenAI Chat model
    llm = ChatOpenAI(model=st.session_state["openai_model"], temperature=0.1)

    # Create the SQL query chain
    generate_query = create_sql_query_chain(llm, db)

    # Initialize Snowpark session
    session = Session.builder.configs(connection_parameters).create()

    # Define the data dictionary
    data_dictionary = {
    "CONVERSATION_ID": "The unique identifier is assigned when a patient initiates a consultation by selecting a symptom category.",
    "CONSULTATION_ID": "The unique key linking an event in the Kena app to Clinic consultation data. It is created once the patient first engages with a clinician.",
    "CREATED_AT": "Timestamp when the conversation was initiated upon patient's symptom category selection.",
    "ENDED_AT": "Timestamp when the conversation concluded, either by clinician clicking the end consultation button or automatic system closure.",
    "QUEUE_JOINED_AT": "Timestamp marking when the patient joined the clinician queue.",
    "QUEUE_ENDED_AT": "Timestamp marking when the patient was picked from the clinician queue by a clinician.",
    "QUEUE_DURATION": "Duration of time the patient spent in the clinician queue. Specific to the queue step before the clinician indicated in that row.",
    "QUEUE_DURATION_IN_SECONDS": "Duration of time the patient spent in the clinician queue, in seconds.",
    "CONSULT_DURATION": "Total duration of time of the consultation. This is measured from the creation of the conversation until either the clinician ends the conversation or the system closes the conversation after 24 hours.",
    "CONSULT_DURATION_IN_SECONDS": "Total duration of time of the consultation.",
    "SNOOZE_DURATION": "Total time for all snoozes within a consultation.",
    "CONSULT_DURATION_MINUS_SNOOZE": "Total duration of the consultation after removing the snooze duration.",
    "SNOOZE_COUNT": "The total number of times a consultation was snoozed for.",
    "CLINICIAN_START_AT": "Timestamp when the clinician picked the patient from the queue and initiated contact with the patient.",
    "CLINICIAN_ENDED_AT": "Timestamp when the clinician concluded the interaction with the patient.",
    "CLINICIAN_DURATION": "Duration of time of the row-specific clinician in a consultation.",
    "AVERAGE_SNOOZE_PER_CLINICIAN": "The average snooze duration per clinician in a consultation.",
    "CLINICIAN_DURATION_MINUS_SNOOZE": "Duration of time of the row-specific clinician in a consultation after removing the average snooze duration per clinician.",
    "CLINICIAN_DURATION_IN_SECONDS": "Duration of time of the row-specific clinician in a consultation.",
    "STAFF_NAME": "Name of the clinician who interacted with the patient in the consultation.",
    "STAFF_ROLE": "The Role of the corresponding clinician.",
    "ASSIGNMENT_ORDER": "The order in which the clinician saw the patient.",
    "STAFF_VIEWS": "The total number of clinicians that were assigned to the consultations, all transfers included.",
    "CONVERSATION_TYPE": "Mode of consultation.",
    "CLINICIAN_ROLES_IN_CONSULT": "The roles of all clinicians in the consultation.",
    "CATEGORY": "Symptom category tile selected at start of consultation.",
    "RATED_AT": "Timestamp for when the patient completed their rating on Kena app.",
    "RATING": "Patient's rating for the overall consultation (positive or negative).",
    "KENA_USER_ID": "Unique identifier for the patient within Kena users database.",
    "PATIENT_GENDER": "Gender of the patient.",
    "PATIENT_AGE": "Age of the patient at time of consultation.",
    "AGE_CATEGORY": "Age category the patient falls in.",
    "PRIMARY_ICD10_CODE": "The main symptom the patient is consulting for. The first ICD10 code diagnosis the patient receives.",
    "REFERRAL_LETTER": "The referral letter sent to patients within the consultation.",
    "REFERRAL_DOCUMENT_ISSUED": "This field shows whether or not a referral document was issued by the respective clinician.",
    "REFERRAL_CATEGORY": "The category type of the referral i.e. whether it’s an investigation emergency etc.",
    "REFERRAL_SECTOR": "The sector to which the patient is referred to is a dropdown list of the following options: public, private, patient_choice.",
    "REFERRAL_TYPE": "The type of referral selected from the following choices: Radiology, Pathology, specialist, Private Hospital, allied, GP, Other.",
    "REFERRAL_SUBTYPE": "These are the subtypes of the specified referral type.",
    "SICK_NOTE_DOCUMENT_ISSUED": "This field shows whether or not a sick note document was issued by the respective clinician.",
    "SCRIPT_DOCUMENT_ISSUED": "This field shows whether or not a prescription (script) note was issued by the respective clinician.",
    "ATTENDANCE_CERTIFICATE_DOCUMENT_ISSUED": "This field shows whether or not an attendance certificate was issued to the patient by the respective clinician.",
    "REGISTRATION_STATUS": "Indicates whether the patient completed the registration process in full.",
    "INVOICE_ID": "Identifier for the consultation's invoice.",
    "INVOICE_CREATED_AT": "Timestamp when the invoice was generated.",
    "AMOUNT": "The total amount the consultation was invoiced for.",
    "CONSULTATION_TYPE": "Type of consultation.",
    "BILLED_CATEGORY": "Whether the consultation was billed or not.",
    "NO_CHARGE_REASON": "Drop down selector for clinicians to indicate reason for no charge.",
    "NOTE": "Free text notes associated with the no charged invoice.",
    "TRANSACTION_ID": "Identifier for the transaction that occurred against a consultation.",
    "TRANSACTION_CREATED_AT": "Timestamp when the invoice was paid and the transaction was recorded.",
    "PAYMENT_METHOD": "The method of payment used to settle invoice.",
    "PAYMENT_STATUS": "The status of the transaction i.e. whether the transaction failed or was a success.",
    "PROMOTION_CUSTOMER_NAME": "The name of the promotion customer.",
    "PROMOTION_NAME": "The name of the promotion run by the partner.",
    "PROMO_CODE": "The promotion code provided to customers to insert at consultation payment.",
    "USER_ALLOCATED_VOUCHER_PROMO_CODE": "The 16 digit voucher code allocated to a user for discount voucher redemption on the app.",
    "PROMOTION_ID": "The unique promotion identifier.",
    "CALL_IN_CONVERSATION": "Boolean indicating whether a call occurred during the consultation.",
    "PATIENT_CANCELLED": "Boolean indicating whether the patient cancelled the consultation.",
    "CLOSE_REASON": "Reason provided for the cancellation of the consultation."
    }

    # Convert data dictionary to a string format
    # data_dictionary_str = json.dumps(data_dictionary, indent=2)
    # Read data dictionary from a text file
    with open('data_dictionary.txt', 'r') as file:
        data_dictionary_str = file.read()

    # prompt = "Please provide me with a complete statistical summary of the queue durations in minutes for each clinician role for the month of July 2024."

    # Create the question with the data dictionary
    question = f"""Using the following data dictionary: {data_dictionary_str}, 
        Question: {prompt} ,
    Instructions: 
    Look in table 'dim_kena__patient_visit_report', there are duplications in the table. Account for duplications by counting distinct records in queries. 
    Identify team consultations as those involving more than one staff member.
    Consider the order of clinicians to determine consultation transfers.
    Use the 'created_at' field for date-related queries.
    Utilize multiple CTEs in your query as needed.
    Ensure accurate joins between CTEs by using common fields. If no common fields exist, prioritize the CTE with the most insights. """

    # Generate the SQL query
    sql_query = generate_query.invoke({"question": question})

    # Remove the semicolon from the end of the generated query string
    sql_query = sql_query.rstrip(';')

    # Display the generated SQL query for debugging purposes
    # st.write("Generated SQL Query:")
    # st.markdown(f"```sql\n{sql_query}\n```")

    # Function to execute SQL query and handle errors
    def execute_query(query, session):
        try:
            result = session.sql(query).to_pandas()
            return result, None
        except Exception as e:
            return None, str(e)

    # Initialize variables for the feedback loop
    retry = True
    max_attempts = 3
    attempt = 0

    while retry and attempt < max_attempts:
        attempt += 1

        # Generate the SQL query
        sql_query = generate_query.invoke({"question": question})

        # Remove the semicolon from the end of the generated query string
        sql_query = sql_query.rstrip(';')

        # Execute the generated SQL query
        result, error = execute_query(sql_query, session)
        
        if error:
            # Feedback to LLM to fix the query
            prompt = f"Error in SQL query: {error}. Please fix the query and try again."
            st.session_state.messages.append({"role": "assistant", "content": prompt})
        else:
            retry = False

    # Handle the case where max attempts are reached
    if attempt == max_attempts and error:
        chain_result = f"Failed to execute query after {max_attempts} attempts. Error: {error}"
    else:
        # Rephrasing Answers for Enhanced Clarity
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question. In your answer reflect the results of the query. Leave out the instructions given in the question in your response.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
        )

        rephrase_answer = answer_prompt | llm | StrOutputParser()

        chain = (
            RunnablePassthrough.assign(query=generate_query).assign(
                result=itemgetter("query") | QuerySQLDataBaseTool(db=db)
            )
            | rephrase_answer
        )

        chain_result = chain.invoke({"question": question})

    # Print the rephrased answer for debugging purposes
    # print("Rephrased Answer:", chain_result)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write("Generated SQL Query:")
        st.markdown(f"```sql\n{sql_query}\n```")
        st.write("Query Results:")
        st.write(result)
        st.write("Rephrased Answer:")
        st.write(chain_result)

    
    st.session_state.messages.append({"role": "assistant", "content": str(sql_query)})
    st.session_state.messages.append({"role": "assistant", "content": str(chain_result)})

    # Check if the session state should be cleared
    if "clear_chat" not in st.session_state:
        st.session_state.clear_chat = True

    # Clear chat history on rerun or page refresh
    if st.session_state.clear_chat:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Kena PVR chatbot. How can I assist you today?"}]
        st.session_state.clear_chat = False

    # Close the session
    session.close()