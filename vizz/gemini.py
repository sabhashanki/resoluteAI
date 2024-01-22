from google.generativeai import GenerativeModel, configure
import streamlit as st
from PIL import Image
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
import os
load_dotenv()
configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title = 'Gemini Testing', layout = 'wide')
col1, col2 = st.columns(2)

def gemini_text(query):
	model = GenerativeModel('gemini-pro')
	response = model.generate_content(query)
	st.write(response.text)

def gemini_vision(img, query):
	model = GenerativeModel('gemini-pro-vision')
	response = model.generate_content([img, query])
	response.resolve()
	st.write(response.text)

def main():
	with col1:
		image_file = st.file_uploader('Upload the image', type = ['jpeg', 'png'])
		image_query = st.text_area('Enter the query for the image?')
		submit_img = st.button('Generate Story')
		if submit_img:
			st.image(image_file)
			img = Image.open(image_file)
			gemini_vision(img, image_query)

			
	with col2:
		text_query = st.text_area('Enter the query for the text?')
		submit_txt = st.button('Answer my query')
		if submit_txt:
			gemini_text(text_query)
	
	
if __name__ == '__main__':
	main()