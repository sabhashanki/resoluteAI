import yaml
import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from streamlit_extras.grid import grid

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

    if authenticator.register_user('Register user', preauthorization=False):
        st.success('User registration successfully')

    with open('./config.yaml', 'a') as file:
        yaml.dump(config, file, default_flow_style=False)

if __name__ == '__main__':
    main()