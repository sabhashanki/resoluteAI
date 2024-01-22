import streamlit as st

class Banking():
	defaultAccNumber = 1

	def __init__(self, name, balance=0):
		self.name = name
		self.balance = balance
		self.accountNumber = Banking.defaultAccNumber
		Banking.defaultAccNumber += 1

	def getBalance(self):
		return self.balance

	def deposit(amount):
		self.balance += amount

	def withdraw(amount):
		if self.balance > amount:
			self.balance -+ amount
		else:
			print('Insufficient Balance')
def main():
	acc_details = []
	st.title('Welcome to ATS Banking')
	acc_no = st.number_input('Enter your account number')
	check_btn = st.button('Login')
	if check_btn:
		for acc in acc_details:
			if acc_no == acc.accountNumber:
				obj = acc
		st.write(f'Welcome {acc.name}')
	newAcc_btn = st.button('New Account')
	name = st.text_area('Enter the name')
	if newAcc_btn:
		new_obj = Banking(name)
	

if __name__ == '__main__':
	main()
