from dotenv import load_dotenv
import streamlit as st
import pdf2image
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
load_dotenv()


# Initialisation
st.set_page_config(layout = 'wide', page_title = 'ATS Tracking System')
st.header('ATS Tracking System')
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(prompt,job_text):
	model=genai.GenerativeModel('gemini-pro')
	response=model.generate_content([prompt,job_text])
	return response.text


def pdf_extract(pdf):
	if pdf is not None:
		text = ""
		pdf_reader = PdfReader(pdf)
		for page in pdf_reader.pages:
			text += page.extract_text()
		return text
	else:
		raise FileNotFoundError('No file uploaded')


def main():
	analyse_prompt = """
	 Act like an experienced and skilled Application Tracking System in Artificial Intelligence domain. 
	 You need to filter out important keywords and skills mentioned in job description with high accuracy
	 {{
	 "Important keywords and skills" : []"
	 }}
	"""
	job_description = st.text_area("Paste Job Description", key = 'input')
	uploaded_file = st.file_uploader('Upload the resume(PDF)', type = ['pdf'])
	analyse_btn = st.button('Analyse my Resume')

	

	if analyse_btn:
		text = pdf_extract(uploaded_file)
		response = get_gemini_response(analyse_prompt, job_description)
		st.subheader(response)
		# for key, value in response.items():
		# 	st.subheader(f'{key} : {value}')



if __name__ == '__main__':
	main()