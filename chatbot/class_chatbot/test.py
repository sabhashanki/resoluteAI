import streamlit as st
from streamlit import query_params
import streamlit_book as stb
import warnings
warnings.filterwarnings("ignore")

def main():
	# Streamlit webpage properties
	st.set_page_config()

	# Streamit book properties
	stb.set_book_config(menu_title="Chatbot",
	                    menu_icon="lightbulb",
	                    options=[
	                            "Chatbot",
	                            "About",
	                            ],
	                    paths=[
	                          "about.py", # single file
	                          "pages/01 Multitest", # a folder
	                          ],
	                    icons=[
	                          "code",
	                          "robot",
	                          ],
	                    save_answers=True,
	                    )
	# second_btn = st.button('go to second page')
	# if second_btn:
	# 	st.switch_page("pages/about.py")


def second():
	st.title('second page')

if __name__ == '__main__':
	main()