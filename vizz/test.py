import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

load_dotenv() ## load all our environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_repsonse(input_csv, prompt, query):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input_csv, prompt, query])
    return response.text

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#Prompt Template

prompt="""
Act like a python matplotlib visualization tool and 
create interesting and insightfull visualization 
based on the user query
user query = {user_query}
Display the output as visualization graph
"""
query = "generate python code for creating bar chart using the csv"
## streamlit app
st.title("Smart Visualization")
input_csv=st.file_uploader("Upload Your Resume",type="csv",help="Please uplaod the CSV")

submit = st.button("Submit")

if submit:
    if input_csv is not None:
        response=get_gemini_repsonse(input_csv, prompt, query)
        st.subheader(response)