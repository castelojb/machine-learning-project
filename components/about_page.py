import streamlit as st
from PIL import Image
import webbrowser
from components.text_content import about_page

class AboutPage:

	git_hub = 'https://github.com/castelojb/machine-learning-project.git'

	text_about = about_page

	def __call__(self):
		st.markdown(self.text_about)

		page_l, page_c, page_r = st.columns(3)

		with page_c:
			git_hub_sample = f"""<div style='text-align: center;'><a href="{self.git_hub}">Git Hub</a></div>"""
			st.markdown(git_hub_sample, unsafe_allow_html=True)




