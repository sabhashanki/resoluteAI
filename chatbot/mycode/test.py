from streamlit_extras.switch_page_button import switch_page
import streamlit as st


def main():
	second_btn = st.button('go to second page')
	if second_btn:
		switch_page("login")


def login():
	st.title('second page')

if __name__ == '__main__':
	main()