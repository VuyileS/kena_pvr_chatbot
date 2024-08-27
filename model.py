import streamlit as st
import openai
import os
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from snowflake.snowpark import Session

# Streamlit app title
st.set_page_config(page_title="Snowflake PVR SQL Querying with LangChain", page_icon="https://cdn.prod.website-files.com/62178119fe77c33f2b5d1ccc/64b908479cae7ff427f6abb6_Header%20Logo.svg")
# Title with image
st.markdown(
    """
    <h1 style="display:flex; align-items:center;">
        <img src="https://cdn.prod.website-files.com/62178119fe77c33f2b5d1ccc/64b908479cae7ff427f6abb6_Header%20Logo.svg" width="40" style="margin-right:10px"/>
        Snowflake (Kena PVR) SQL Querying
    </h1>
    """,
    unsafe_allow_html=True
)
sidebar_style = """
    <style>
    /* Targeting the sidebar more specifically */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #eb345e !important;
    }
    /* Attempting to target all text within the sidebar to change its color to white */
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .st-cb, [data-testid="stSidebar"] .st-dd, [data-testid="stSidebar"] {
        color: #ffffff !important;
    }

    /* If the above doesn't cover all text, this broader rule might help */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    </style>
    """
# Sidebar login section
st.sidebar.subheader("Login to Snowflake")
st.markdown(sidebar_style, unsafe_allow_html=True)
snowflake_username = st.sidebar.text_input("Snowflake Username", key="username")
snowflake_password = st.sidebar.text_input("Snowflake Password", type="password", key="password")

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
)

if not snowflake_username or not snowflake_password:
    st.sidebar.warning("Please enter your Snowflake credentials to log in.")
    st.stop()

if not openai_api_key:
    st.sidebar.warning("Please add your OpenAI API key to continue.")
    st.stop()

# Add a dropdown to select the GPT model
gpt_model = st.sidebar.selectbox(
    "Choose GPT Model",
    ["gpt-3.5-turbo", "gpt-4"]
)

# Model information in the sidebar
st.sidebar.markdown("""
### Model Information
- **GPT-3.5-Turbo**: Fast and efficient, suitable for most queries with a token limit of 4096.
- **GPT-4**: More powerful, handles complex queries better, with a token limit of 8192, but may be slower and more costly.
""")

# Establish Snowflake connection
connection_parameters = {
    "account": os.getenv('snowflake_account'),
    "user": snowflake_username,
    "password": snowflake_password,
    "warehouse": os.getenv('snowflake_warehouse'),
    "database": os.getenv('snowflake_database'),
    "schema": os.getenv('snowflake_schema')
}

try:
    session = Session.builder.configs(connection_parameters).create()
    st.sidebar.success("Login successful!")
except Exception as e:
    st.sidebar.error(f"Login failed: {e}")
    st.stop()

# Setup ChatOpenAI LLM with the selected model
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.1,
    streaming=True,
    model_name=gpt_model
)

# Setup Snowflake DB with SQLAlchemy
snowflake_url = f"snowflake://{snowflake_username}:{snowflake_password}@{os.getenv('snowflake_account')}/{os.getenv('snowflake_database')}/{os.getenv('snowflake_schema')}?warehouse={os.getenv('snowflake_warehouse')}"
engine = create_engine(snowflake_url)
db = SQLDatabase(engine, include_tables=["dim_kena__patient_visit_report"])

# Create a toolkit for the SQL agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create SQL agent with verbose output and callback handling
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True  
)

# Simplified context from the data dictionary
context_str = """
dim_kena__patient_visit_report contains fields such as:
- CONVERSATION_ID: Unique identifier assigned when a patient initiates a consultation.
- CONSULTATION_ID: Key linking an event in the Kena app to Clinic consultation data.
- CREATED_AT: Timestamp when the conversation was initiated upon patient's symptom category selection.
- ENDED_AT: Timestamp when the conversation concluded, either by clinician or system closure.
- QUEUE_JOINED_AT: Timestamp marking when the patient joined the clinician queue.
- QUEUE_ENDED_AT: Timestamp marking when the patient was picked from the clinician queue by a clinician.
- QUEUE_DURATION: Duration of time the patient spent in the clinician queue.
- QUEUE_DURATION_IN_SECONDS: Duration of time the patient spent in the clinician queue, in seconds.
- CONSULT_DURATION: Total duration of time of the consultation.
- CONSULT_DURATION_IN_SECONDS: Total duration of time of the consultation, in seconds.
- SNOOZE_DURATION: Total time for all snoozes within a consultation.
- CONSULT_DURATION_MINUS_SNOOZE: Total duration of the consultation after removing the snooze duration.
- SNOOZE_COUNT: The total number of times a consultation was snoozed.
- CLINICIAN_START_AT: Timestamp when the clinician picked the patient from the queue and initiated contact.
- CLINICIAN_ENDED_AT: Timestamp when the clinician concluded the interaction with the patient.
- CLINICIAN_DURATION: Duration of time of the row-specific clinician in a consultation.
- AVERAGE_SNOOZE_PER_CLINICIAN: The average snooze duration per clinician in a consultation.
- CLINICIAN_DURATION_MINUS_SNOOZE: Duration of time of the row-specific clinician after removing the average snooze duration per clinician.
- CLINICIAN_DURATION_IN_SECONDS: Duration of time of the row-specific clinician in a consultation, in seconds.
- STAFF_NAME: Name of the clinician who interacted with the patient.
- STAFF_ROLE: The role of the corresponding clinician.
- ASSIGNMENT_ORDER: The order in which the clinician saw the patient.
- STAFF_VIEWS: The total number of clinicians assigned to the consultations.
- CONVERSATION_TYPE: Mode of consultation.
- CLINICIAN_ROLES_IN_CONSULT: The roles of all clinicians in the consultation.
- CATEGORY: Symptom category tile selected at the start of the consultation.
- RATED_AT: Timestamp for when the patient completed their rating on the Kena app.
- RATING: Patient's rating for the overall consultation (positive or negative).
- KENA_USER_ID: Unique identifier for the patient within the Kena users database.
- PATIENT_GENDER: Gender of the patient.
- PATIENT_AGE: Age of the patient at the time of consultation.
- AGE_CATEGORY: Age category the patient falls in.
- PRIMARY_ICD10_CODE: The main symptom the patient is consulting for.
- REFERRAL_LETTER: The referral letter sent to patients within the consultation.
- REFERRAL_DOCUMENT_ISSUED: Whether or not a referral document was issued.
- REFERRAL_CATEGORY: The category type of the referral.
- REFERRAL_SECTOR: The sector to which the patient is referred.
- REFERRAL_TYPE: The type of referral selected.
- REFERRAL_SUBTYPE: The subtypes of the specified referral type.
- SICK_NOTE_DOCUMENT_ISSUED: Whether or not a sick note document was issued.
- SCRIPT_DOCUMENT_ISSUED: Whether or not a prescription note was issued.
- ATTENDANCE_CERTIFICATE_DOCUMENT_ISSUED: Whether or not an attendance certificate was issued.
- REGISTRATION_STATUS: Indicates whether the patient completed the registration process in full.
- INVOICE_ID: Identifier for the consultation's invoice.
- INVOICE_CREATED_AT: Timestamp when the invoice was generated.
- AMOUNT: The total amount the consultation was invoiced for.
- CONSULTATION_TYPE: Type of consultation.
- BILLED_CATEGORY: Whether the consultation was billed or not.
- NO_CHARGE_REASON: Reason for no charge if applicable.
- NOTE: Free text notes associated with the no-charge invoice.
- TRANSACTION_ID: Identifier for the transaction that occurred against a consultation.
- TRANSACTION_CREATED_AT: Timestamp when the transaction was recorded.
- PAYMENT_METHOD: The method of payment used to settle the invoice.
- PAYMENT_STATUS: The status of the transaction.
- PROMOTION_CUSTOMER_NAME: The name of the promotion customer.
- PROMOTION_NAME: The name of the promotion run by the partner.
- PROMO_CODE: The promotion code provided to customers at consultation payment.
- USER_ALLOCATED_VOUCHER_PROMO_CODE: The 16-digit voucher code allocated to a user for discount voucher redemption.
- PROMOTION_ID: The unique promotion identifier.
- CALL_IN_CONVERSATION: Whether a call occurred during the consultation.
- PATIENT_CANCELLED: Whether the patient canceled the consultation.
- CLOSE_REASON: Reason provided for the cancellation of the consultation.
"""

# Chat UI
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your PVR queries today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything about the patient visit report!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Create the full prompt with context
    prompt_template = f"""
    Using the following context into the dim_kena__patient_visit_report table, answer prompts in a contextual manner: {context_str}

    Question: {{query}}
    Instructions: 
    Please add analytics.prod. before dim_kena__patient_visit_report when query the dim_kena__patient_visit_report table to as a schema reference.
    Look in table 'dim_kena__patient_visit_report', there are duplications in the table. Account for duplications by counting distinct records in queries that require counts. 
    Do not use backticks or triple backticks to format SQL code.
    Please ensure that Instead of using backticks (```) when executing and writing your queries use double quotes (").
    Identify team consultations as those involving more than one staff member.
    Consider the order of clinicians to determine consultation transfers.
    Use the 'created_at' field for date-related queries.
    Utilize multiple CTEs in your query when needed. 
    Ensure accurate joins between CTEs by using common fields. If no common fields exist, prioritize the CTE with the most insights.
    When asked about queue durations and clinician durations use the QUEUE_DURATION_IN_SECONDS and CLINICIAN_DURATION_IN_SECONDS respectively which are in seconds, always convert to minutes.
    """

    modified_query = prompt_template.format(query=user_query)

    # Remove backticks if they were mistakenly added
    cleaned_query = modified_query.replace("```sql", "").replace("```", "").strip()

    # Ensure the prompt and completion don't exceed the token limit
    # Ensure the prompt and completion don't exceed the token limit
    max_token_limit = 8192 if gpt_model == "gpt-4" else 4096
    if len(cleaned_query.split()) + 256 > max_token_limit:
        st.error("The query is too long for the selected model. Please reduce the query or context size.")
    else:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            response = agent.run(cleaned_query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

# Optionally, close the session when the app exits
if st.sidebar.button("Log out"):
    if 'session' in st.session_state:
        st.session_state['session'].close()
    st.sidebar.success("Logged out.")
    st.session_state.clear()
    st.write('<meta http-equiv="refresh" content="0; url=/" />', unsafe_allow_html=True)
