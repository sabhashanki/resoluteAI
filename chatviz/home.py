from openai import OpenAI
import streamlit as st
import pandas as pd
import re
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
load_dotenv()
st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> Chat2VIS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations using Natural Language with ChatGPT and Code Llama</h2>", unsafe_allow_html=True)


def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]



if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("beacon_data.csv")
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]


with st.sidebar:
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()
    # Add facility to upload a dataset
    uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
    # When we add the radio buttons we want to default the selection to the first
    index_no = 0
    # if uploaded_file:
    #     # Read in the data, add it to the list of available datasets. Give it a nice name.
    #     file_name = uploaded_file.name[:-4].capitalize()
    df = pd.read_csv(uploaded_file)
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:", datasets.keys(), index=index_no)

st.dataframe(df, hide_index=True)


question = st.text_area(":eyes: What would you like to visualise?", height=2)
go_btn = st.button("Go…")

code_prompt = """
            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """

if go_btn:
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system", "content":code_prompt},
         {"role":"user", "content":question}])
    st.write(response.choices[0].message.content)
    code = extract_python_code(response.choices[0].message.content)
    st.write(code)
    # exec(code)
    llm = ChatOpenAI(model="gpt-4")
    pandas_df_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                return_intermediate_steps=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=False,
            )

    answer = pandas_df_agent(question)
    if answer["intermediate_steps"]:
        action = answer["intermediate_steps"][-1][0].tool_input["query"]
        st.write(f"Executed the code ```{action}```")