import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from streamlit.logger import get_logger
import streamlit as st
from yaml.loader import SafeLoader
import yaml
import os
logger = get_logger(__name__)
def profile():
    logger.info('accessing profile section')
    if st.session_state["authentication_status"]:
        profile_menu = option_menu(None, ["Update Password", "Update Profile"],
                                   icons=['unlock', 'lock'],
                                   orientation="horizontal", key='menu_profile')
        if profile_menu == 'Update Password':
            change_pass()
        if profile_menu == 'Update Profile':
            update_profile()
    if not st.session_state["authentication_status"]:
        st.subheader('Login to access the profile')

def home():
    logger.info('accessing profile section')
    if not st.session_state["authentication_status"]:
        home_menu = option_menu(None, ["Login", "Register", "Forgot Password"],
                                icons=['unlock', 'lock', 'lock'],
                                orientation="horizontal", key='home_menu')
        if home_menu == 'Login':
            login()
        if home_menu == 'Register':
            register()
        if home_menu == 'Forgot Password':
            forgot_pass()

    if st.session_state["authentication_status"]:
        st.subheader(f'Welcome *{st.session_state["name"]}*, Click on the chat !!')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")


# Login Page
def login():
    logger.info('accessing login section')
    st.session_state.authenticator.login('Login', location='main')
    if 'login_key' not in st.session_state:
        st.session_state.login_key = None
    if st.session_state["authentication_status"]:
        logger.info('Login authentication successful')
        user_name = st.session_state["name"]
        parent = os.getcwd()
        st.session_state.user_path = os.path.join(parent, user_name)
        if not os.path.exists(st.session_state.user_path):
            os.mkdir(st.session_state.user_path)
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        logger.info('Incorrect password/username')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


# Register Page
def register():
    if st.session_state.authenticator.register_user('Registration', preauthorization=False):
        st.success('User registration successfully')
        logger.info('Registration successful')
    with open('./config.yaml', 'a') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)


# create onetime password
def forgot_pass():
    username_forgot_pw, email, random_password = st.session_state.authenticator.forgot_password(
        'Create one-time password')
    if username_forgot_pw:
        st.success(f'New random password is : {random_password}.. Change it in next login')
        logger.info(f'Onetime password generated for the user {username_forgot_pw}: {random_password}')
    elif not username_forgot_pw:
        st.warning('Enter the username')
        logger.info('Invalid username for generating onetime password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)
        logger.info('onetime password updated in yaml file')


# update password
def change_pass():
    if st.session_state["authentication_status"]:
        if st.session_state.authenticator.reset_password(st.session_state["username"], 'Update Password'):
            st.success('New password changed')
            logger.info('user password changed')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to change the password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)
        logger.info('new password updated in yaml file')


# update profile info
def update_profile():
    if st.session_state["authentication_status"]:
        if st.session_state.authenticator.update_user_details(st.session_state["username"], 'Update Profile'):
            st.success('Entries updated successfully')
            logger.info('user info changed successfully')
        with st.sidebar:
            st.session_state.authenticator.logout("Logout", "sidebar")
    if not st.session_state["authentication_status"]:
        st.subheader('You need to login to update the profile')
    with open('./config.yaml', 'a') as file:
        yaml.dump(st.session_state.config, file, default_flow_style=False)
        logger.info('user info updated in yaml file')

def main():
	with open('./config.yaml') as file:
		st.session_state.config = yaml.load(file, Loader=SafeLoader)

	st.session_state.authenticator = stauth.Authenticate(
		st.session_state.config['credentials'],
		st.session_state.config['cookie']['name'],
		st.session_state.config['cookie']['key'],
		st.session_state.config['cookie']['expiry_days'],
		st.session_state.config['preauthorized']
	)
	tab1, tab2, tab3 = st.tabs(["Home", "Profile", "Forgot Password"])

	with tab1:
	   st.header("Home")
	   home()

	with tab2:
		st.header("Profile")
		profile()

	with tab3:
	   st.header("Forgot Password")
	   forgot_pass()

if __name__ == "__main__":
    main()