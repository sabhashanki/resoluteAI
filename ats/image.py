import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
st.set_page_config(page_title = 'Image Captioning', layout = 'wide')
prompt = """
Act like a artificial intelligence image examiner to 
study the image and provide information about the image like 
objects in the image, color and location with great accuracy

Give response like the format mentioned below
{{
	"Objects in the picture : []",
	"Dress Outfit and color" : ""
}}
"""

prompt2 = """
Act like a artificial intelligence image examiner to 
study the image and provide troll and funny comments about the outfit
Give the response as a single string
"""

def get_gemini_repsonse(input, prompt):
	model = genai.GenerativeModel('gemini-pro-vision')
	response = model.generate_content([input, prompt])
	return response.parts

def main():
	st.title('Image Captioning')
	uploaded_pic = st.file_uploader('Upload the picture and chat with it', type=['jpeg', 'png'])
	submit_btn = st.button('Submit')

	if submit_btn:
		if uploaded_pic is not None:
			st.image(uploaded_pic)
			img = Image.open(uploaded_pic)
			response = get_gemini_repsonse(img, prompt)
			st.subheader(response)


if __name__ == '__main__':
	main()