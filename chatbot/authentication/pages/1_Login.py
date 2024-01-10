import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
st.set_page_config(page_title='Login')
           
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
        with st.sidebar:
            authenticator.logout("Logout", "sidebar")
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.title('Some content')

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')

    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

if __name__ == '__main__':
    main()