import streamlit as st
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml

def home():
    st.title('This is the home page !!!!')


def login():
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
    # if st.session_state["authentication_status"]:
    #     st.title(f'Welcome *{st.session_state["name"]}*')
    #     st.subheader('Click on the Chat to upload document and access AI chatbot')
    #     with st.sidebar:
    #            authenticator.logout("Logout", "sidebar")
    # elif st.session_state["authentication_status"] is False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state["authentication_status"] is None:
    #     st.warning('Please enter your username and password')


def register():
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
    

def about():
    st.title('This is the about page !!!!')
    file_folder = st.file_uploader("Upload the document",type=['pdf','txt','md','rst'],accept_multiple_files=True)           
    process_button=st.button("Create Embeddings")


def main():

    with st.sidebar:
        doc_type = st.selectbox('Select document type', 
            (
                'None',
                'PDF',
                'TXT',
                'RST',
                'MD'
             )
            )
        btn_home = st.button('Home')
        btn_login = st.button('Login')
        btn_register = st.button('Register')
        btn_about = st.button('About')

    if btn_login:
        st.session_state.runpage = login
        st.session_state.runpage()
        st.rerun

    if btn_register:
        st.session_state.runpage = register
        st.session_state.runpage()
        st.rerun

    if btn_about:
        st.session_state.runpage = about
        st.session_state.runpage()
        st.rerun

    if btn_home:
        st.session_state.runpage = home
        st.session_state.runpage()
        st.rerun


if __name__ == '__main__':
    main()