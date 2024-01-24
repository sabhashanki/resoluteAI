import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import yaml
load_dotenv()
st.set_page_config(layout = 'centered')
text = """
A chatbot is a software or computer program that simulates human conversation or "chatter" through text or voice interactions.Users in both business-to-consumer (B2C) and business-to-business (B2B) environments increasingly use chatbot virtual assistants to handle simple tasks. Adding chatbot assistants reduces overhead costs, uses support staff time better and enables organizations to provide customer service during hours when live agents aren't available."""

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
	if st.session_state["authentication_status"]:
		with st.sidebar:
	           authenticator.logout("Logout", "sidebar")
	my_grid = grid(1, 1, 2, vertical_align="bottom")
	my_grid.title('Welcome to the Home Page')
	my_grid.subheader(text)
	loginBtn = my_grid.button("Login", use_container_width=False)
	regisBtn = my_grid.button("Register", use_container_width=False)
	if loginBtn:
		st.switch_page("/pages/login.py")
	if regisBtn:
		st.switch_page("/pages/register.py")

def second():
	st.title('second page')

if __name__ == '__main__':
	main()