import streamlit as st
import streamlit_authenticator as stauth
import yaml
import os
from utils import add_company_logo
from streamlit_option_menu import option_menu
from yaml.loader import SafeLoader
from st_pages import Page, show_pages

def home():
    st.title("This is my Home page")

def login():
    st.title("Login page")
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    authenticator.login('Login', location = 'main')
    if st.session_state["authentication_status"]:
        st.title(f'Welcome *{st.session_state["name"]}*')
        st.subheader('Click on the Chat to upload document and access AI chatbot')
        user_name = st.session_state["name"]
        parent = os.getcwd()
        path = os.path.join(parent, user_name)
        if not os.path.exists(path):
            os.mkdir(path)
        with st.sidebar:
               authenticator.logout("Logout", "sidebar")
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


def main():
    add_company_logo()

    if 'key' not in st.session_state:
        st.session_state.key = None
    with st.sidebar:
        st.session_state.key = option_menu("ChatBot", ["Home", 'Login'], 
            icons=['house', 'unlock'], menu_icon="chat", orientation='vertical')

    
    st.markdown("# Welcome to Chatbot Main page 🎈")
    # show_pages([Page("home.py", "Home"),
    #             Page("register.py", "Register"), 
    #             Page("login.py", "Login"), 
    #             Page("chat.py", "Chat", in_section=False),
    #             Page("update_profile.py", "Update Profile"), 
    #             Page("reset_pass.py", "Reset Password"),
    #             Page("forgot_pass.py", "Forgot Password"), 
    #             Page("about.py", "About"), 
    #             ])
    
    if st.session_state.key == 'Home':
        home()
    if st.session_state.key == 'Login':
        login()
if __name__ == '__main__':
    main()