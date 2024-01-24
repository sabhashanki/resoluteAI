import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import os

def main():
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
        user_name = st.session_state["name"]
        parent = os.getcwd()
        path = os.path.join(parent, user_name)
        if not os.path.exists(path):
            os.mkdir(path)
        chatBtn = st.button('Click to Chat')
        if chatBtn:
            st.switch_page('pages/chat.py')
        with st.sidebar:
               authenticator.logout("Logout", "sidebar")
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')

    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


if __name__ == '__main__':
    main()