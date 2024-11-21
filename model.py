import streamlit as st
import openai
import os
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI
# from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.callbacks.base import BaseCallbackHandler
from snowflake.snowpark import Session
import pandas as pd
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
# from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.sql_database import SQLDatabase
from pydantic import BaseModel
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from pydantic import BaseModel


from registrations_geolocation import geolocation_spread
from patient_analysis import patient_messaging
from clinic_registrations_geolocation import clinic_geolocation_spread

st.set_page_config(page_title="Snowflake PVR SQL Querying with LangChain", page_icon="https://cdn.prod.website-files.com/62178119fe77c33f2b5d1ccc/64b908479cae7ff427f6abb6_Header%20Logo.svg")
tab1, tab2, tab3 = st.tabs(["SQL Querying", "Geolocation Spread", "Support Messaging"])#, , tab4 "Clinic Medical Centre"

with tab1:
    # Streamlit app title
    
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
    
    llm_prompt_template = """
    Using the following context into the dim_kena__patient_visit_report table, answer prompts in a contextual manner: {context_str}

        Question: {{query}}
        Instructions: 
        Please add analytics.prod. before dim_kena__patient_visit_report when query the dim_kena__patient_visit_report table to as a schema reference.
        Look in table 'dim_kena__patient_visit_report', there are duplications in the table. Account for duplications by counting distinct records in queries that require counts. 
        Given an input question, create a syntactically correct Snowflake SQL query.
            - Use only double quotes (`"`) if necessary for table or column names.
            - Do not use backticks (` ``` `).
        Do not use backticks or triple backticks to format SQL code or column names.
        Please ensure that Instead of using backticks (```) when executing and writing your queries use double quotes (") because this throws an error when querying in snowflake.
        Identify team consultations as those involving more than one staff member.
        Consider the order of clinicians to determine consultation transfers.
        Use the 'created_at' field for date-related queries.
        Utilize multiple CTEs in your query when needed. 
        Always include the results of your queries in the final answer.
        Ensure accurate joins between CTEs by using common fields. If no common fields exist, prioritize the CTE with the most insights.
        When asked about queue durations and clinician durations use the QUEUE_DURATION_IN_SECONDS and CLINICIAN_DURATION_IN_SECONDS respectively which are in seconds, always convert to minutes.
        When calculating averages, ensure that duplicates in the table are considered appropriately. The average should be calculated based on distinct values only.
        When the results are too long you can write the final answer as a table. 
        
    """
    # Setup ChatOpenAI LLM with the selected model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.1,
        streaming=True,
        prompt_template=llm_prompt_template,
        model_name=gpt_model
    )
    class BaseCache(BaseModel):
        pass

    SQLDatabaseChain.BaseCache = BaseCache

    # Define Callbacks and rebuild SQLDatabaseChain
    class Callbacks(BaseCallbackHandler):
        """Define any necessary logic for callbacks."""
        pass

    SQLDatabaseChain.Callbacks = Callbacks
    SQLDatabaseChain.model_rebuild()
    # Setup Snowflake DB with SQLAlchemy
    snowflake_url = f"snowflake://{snowflake_username}:{snowflake_password}@{os.getenv('snowflake_account')}/{os.getenv('snowflake_database')}/{os.getenv('snowflake_schema')}?warehouse={os.getenv('snowflake_warehouse')}"
    engine = create_engine(snowflake_url)
    db = SQLDatabase(engine, include_tables=["dim_kena__patient_visit_report"])

    
    
    # Initialize ChatOpenAI LLM
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.1,
        model_name=gpt_model
    )
  
    # Set up a Streamlit callback handler
    st_cb = StreamlitCallbackHandler(st.container())
    # Create the SQLDatabaseChain
    # sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, callbacks=[st_cb])
    sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


    

    # Fetch a limited preview of the table data
    query = "SELECT * FROM dim_kena__patient_visit_report LIMIT 10"
    df_limited = pd.read_sql(query, engine)

    # Display a preview of the data in an expander before the first message
    with st.expander("## 🔎 Please preview the patient visit report here"):
        st.dataframe(df_limited)
    
    class CustomStreamlitCallbackHandler(BaseCallbackHandler):
        """Custom callback handler to display meaningful intermediate outputs in the Streamlit app."""

        def __init__(self):
            self.steps = []  # Store steps as a list of tuples (title, content)

        def on_text(self, text: str, **kwargs):
            """Process intermediate outputs dynamically and filter out irrelevant information."""

            # Skip verbose instructions or table schema information
            if "CREATE TABLE" in text or "Question" in text:
                return  # Ignore verbose content containing schema or context
            
            if "SQLQuery:" in text:
                sql_query = text.split("SQLQuery:", 1)[-1].strip()
                # sanitized_query = sql_query.replace("`", "").strip()  # Remove backticks
                sanitized_query = (
                    sql_query.replace("`", "")  # Remove backticks
                    .replace("```", "")         # Remove triple backticks
                    .strip()                    # Trim whitespace
                )
                if sanitized_query:
                    self.steps.append(("Generated SQL Query", f"```sql\n{sanitized_query}\n```"))

                    try:
                        # Execute the sanitized query
                        result = db.run(sanitized_query)
                        self.steps.append(("SQL Query Result", f"```plaintext\n{result}\n```"))
                    except Exception as e:
                        error_message = f"Error executing query: {str(e)}"
                        self.steps.append(("SQL Execution Error", f"```plaintext\n{error_message}\n```"))
            elif "SQLResult:" in text:
                sql_result = text.split("SQLResult:", 1)[-1].strip()
                if sql_result:
                    self.steps.append(("SQL Query Result", f"```plaintext\n{sql_result}\n```"))
            elif "Answer:" in text:
                answer = text.split("Answer:", 1)[-1].strip()
                if answer:
                    self.steps.append(("Final Answer", f"```plaintext\n{answer}\n```"))
            else:
                if text.strip():
                    self.steps.append(("Intermediate Step", f"```plaintext\n{text.strip()}\n```"))

        def get_steps(self):
            """Return all grouped steps."""
            return self.steps




    # Chat UI
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you with your PVR queries today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything about the patient visit report!")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        assistant_message = st.chat_message("assistant")
        response = None  # Ensure response is defined
        with assistant_message:
            with st.spinner("Processing your query..."):
                try:
                    # Initialize the custom callback handler
                    custom_st_cb = CustomStreamlitCallbackHandler()

                    # Combine the context and user query
                    full_input = f"{llm_prompt_template}\n\nQuestion: {user_query}"

                    # Run the SQL chain
                    response = sql_chain.run(full_input, callbacks=[custom_st_cb])

                    # Ensure backticks are removed from any generated queries
                    for step_title, step_content in custom_st_cb.get_steps():
                        if "Generated SQL Query" in step_title:
                            sql_query = step_content.strip("```sql").strip()
                            sanitized_query = sql_query.replace("`", "").strip()
                            
                            # Execute the sanitized query
                            try:
                                result = db.run(sanitized_query)
                                custom_st_cb.steps.append(
                                    ("Sanitized SQL Query Result", f"```plaintext\n{result}\n```")
                                )
                            except Exception as e:
                                custom_st_cb.steps.append(
                                    ("SQL Execution Error", f"```plaintext\n{e}\n```")
                                )

                    # Display intermediate steps
                    for step_title, step_content in custom_st_cb.get_steps():
                        with st.expander(f"🔍 {step_title}", expanded=False):
                            st.markdown(step_content, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    response = f"An error occurred: {e}"  # Provide a fallback response

        # Append the final response to session state and display it
        st.session_state.messages.append({"role": "assistant", "content": response})
        assistant_message.write(response)







    # Optionally, close the session when the app exits
    if st.sidebar.button("Log out"):
        if 'session' in st.session_state:
            st.session_state['session'].close()
        st.sidebar.success("Logged out.")
        st.session_state.clear()
        st.write('<meta http-equiv="refresh" content="0; url=/" />', unsafe_allow_html=True)

with tab2:
    geolocation_spread(snowflake_username,snowflake_password )

with tab3:
    patient_messaging(snowflake_username,snowflake_password )

# with tab4:
#     clinic_geolocation_spread(snowflake_username,snowflake_password )

# with tab5:
#     pre_consult(snowflake_username,snowflake_password )