import streamlit as st
import openai
import os
import json
from sqlalchemy import create_engine, inspect
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_sql_agent
from operator import itemgetter
from snowflake.snowpark import Session

# Streamlit app title
st.title("Kena PVR Chatbot with SQL Integration")

# Initialize session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.snowflake_username = ""
    st.session_state.snowflake_password = ""

# Sidebar login section
with st.sidebar:
    st.subheader("Login to Snowflake")
    st.session_state.snowflake_username = st.text_input("Snowflake Username", key="username")
    st.session_state.snowflake_password = st.text_input("Snowflake Password", type="password", key="password")

    if st.button("Log in"):
        if st.session_state.snowflake_username and st.session_state.snowflake_password:
            # Initialize Snowflake connection parameters
            connection_parameters = {
                "account": os.getenv('snowflake_account'),
                "user": st.session_state.snowflake_username,
                "password": st.session_state.snowflake_password,
                "warehouse": os.getenv('snowflake_warehouse'),
                "database": os.getenv('snowflake_database'),
                "schema": os.getenv('snowflake_schema')
            }

            try:
                # Attempt to establish a Snowflake session
                session = Session.builder.configs(connection_parameters).create()
                st.session_state.logged_in = True
                st.session_state.session = session
                st.sidebar.success("Login successful!")
            except Exception as e:
                st.sidebar.error(f"Login failed: {e}")
        else:
            st.sidebar.warning("Please enter both your username and password.")

# Show the main app only if logged in
if st.session_state.logged_in:
    # Main app functionality after successful login
    session = st.session_state.session

    # Load environment variables

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

        snowflake_url = f"snowflake://{st.session_state.snowflake_username}:{st.session_state.snowflake_password}@{os.getenv('snowflake_account')}/{os.getenv('snowflake_database')}/{os.getenv('snowflake_schema')}?warehouse={os.getenv('snowflake_warehouse')}"

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

        # data_dictionary_str = json.dumps(data_dictionary, indent=2)

        # Read the data dictionary from the .txt file
        with open('data_dictionary.txt', 'r') as file:
            data_dictionary_str = file.read()
        # Convert the data dictionary string to a JSON formatted string
        data_dictionary_json_str = json.dumps(data_dictionary_str, indent=2)


        # Create the question with the data dictionary
        question = f"""Using the following data dictionary: {data_dictionary_json_str}, 
            Question: {prompt} ,
        Instructions: 
        Look in table 'dim_kena__patient_visit_report', there are duplications in the table. Account for duplications by counting distinct records in queries that require counts. 
        Identify team consultations as those involving more than one staff member.
        Consider the order of clinicians to determine consultation transfers.
        Use the 'created_at' field for date-related queries.
        Utilize multiple CTEs in your query when needed. 
        Ensure accurate joins between CTEs by using common fields. If no common fields exist, prioritize the CTE with the most insights.
        When asked about queue durations and clinician durations use the QUEUE_DURATION_IN_SECONDS and CLINICIAN_DURATION_MINUS_SNOOZE respectively which are in seconds, always convert to minutes."""

        # Generate the SQL query
        sql_query = generate_query.invoke({"question": question})

        # Function to execute SQL query and handle errors
        def execute_query(query, session):
            try:
                result = session.sql(query).to_pandas()
                return result, None
            except Exception as e:
                return None, str(e)

        # Execute the generated SQL query
        result, error = execute_query(sql_query, session)

        if error:
            # Handle SQL execution error
            st.error(f"Error in SQL query execution: {error}")
        else:
            # Rephrase and display the result
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

    # Clear chat history on rerun or page refresh
    if "clear_chat" not in st.session_state:
        st.session_state.clear_chat = True

    if st.session_state.clear_chat:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Kena PVR chatbot. How can I assist you today?"}]
        st.session_state.clear_chat = False

    # Logout button
    if st.sidebar.button("Log out"):
        # Close the session before logging out
        if 'session' in st.session_state:
            st.session_state['session'].close()
        
        # Clear session state
        st.session_state.clear()

        # Simulate a page reload by redirecting to an empty URL (this will refresh the app)
        st.write('<meta http-equiv="refresh" content="0; url=/" />', unsafe_allow_html=True)
