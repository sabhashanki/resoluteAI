import streamlit as st
import streamlit_authenticator as stauth
import yaml
from utils import add_company_logo
from yaml.loader import SafeLoader
from st_pages import Page, show_pages

def main():
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

    st.markdown("# Welcome to Chatbot 🎈")
    st.write('This application lets you to chat with PDF and answer all question related to it')
    show_pages([Page("home.py", "Home"),
                Page("login.py", "Login"), 
                Page("register.py", "Register"), 
                Page("reset_pass.py", "Reset Password"),
                Page("forgot_pass.py", "Forgot Password"), 
                Page("update_profile.py", "Update Profile"), 
                Page("about.py", "About"), 
                Page("chat.py", "Chat", in_section=False)
                ])
    

if __name__ == '__main__':
    main()