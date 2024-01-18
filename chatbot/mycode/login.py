import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from utils import add_company_logo
from dotenv import load_dotenv
st.set_page_config(page_title='Login')
import os

def main():
    load_dotenv()
    add_company_logo()

    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    authenticator.login('Login', 'main')

    if st.session_state["authentication_status"]:
        st.title(f'Welcome *{st.session_state["name"]}*')
        st.subheader('Click on the Chat to upload document and access AI chatbot')
        with st.sidebar:
               authenticator.logout("Logout", "sidebar")
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')

    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


if __name__ == '__main__':
    main()