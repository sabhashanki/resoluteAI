from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import pandas as pd
import re


# Initialization
load_dotenv()
st.set_page_config(layout='wide')
st.subheader("Create Matplotlib visualization using ChatGPT and LangChain")
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'prompt' not in st.session_state:
	st.session_state.prompt = None
if 'df' not in st.session_state:
	st.session_state.df = None

def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]


def code_promptInit(df):
	code_prompt = """Generate the code <code> for plotting the pandas dataframe {df} in Matplotlib using Python 3.10,
	            in the format requested. The solution should be given using Matplotlib
	            and only Matplotlib. Return only the code without any comments """
	code_prompt += "\nDataframe has " + str(len(df.columns)) + 'with column names ' + str(df.columns)
	for i in df.columns:
	    if df.dtypes[i] == 0:
	        code_prompt += '\nColumn ' + str(i) + ' has categorical datatype'
	code_prompt += "\n Remaining all columns belong to int64 or float datatype"
	code_prompt += "\nLabel the x and y axes appropriately."
	code_prompt += "\nAdd a title. Set the fig subtitle as empty."
	return code_prompt


def main():
	with st.sidebar:
	    uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
	    submitBtn = st.button('Submit')
	if submitBtn:
		st.session_state.df = pd.read_csv(uploaded_file)
		st.session_state.df.drop_duplicates(inplace = True)
		st.session_state.prompt = code_promptInit(st.session_state.df)
	col1, col2 = st.columns(2)
	with col1:
		try:
			st.dataframe(st.session_state.df, hide_index=True)
		except:
			st.write('Upload CSV to see the dataframe')

		question = st.text_area("What would you like to visualise?", height=2)
		go_btn = st.button("Go…")


	with col2:
		if go_btn:
			client = OpenAI()
			response = client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=[{"role":"system", "content":st.session_state.prompt},
				 {"role":"user", "content":question}])
			code = response.choices[0].message.content
			# code = "import matplotlib.pyplot as plt\n" + str(code)
			# st.write(code)
			df = st.session_state.df
			plot_area = st.empty()
			plot_area.pyplot(exec(code))


if __name__ == '__main__':
	main()








