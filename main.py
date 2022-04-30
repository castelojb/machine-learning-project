import streamlit as st
from PIL import Image

from components.regression_page import RegressionPage

st.set_page_config(
    page_title="Painel ML",
    layout="wide",
)

# hide_streamlit_style = """
#
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>
#
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

from components import AboutPage


about_card = AboutPage()

regression_page = RegressionPage()

if __name__ == '__main__':

	st.markdown("<h1 style='text-align: center;'>Painel ML</h1>", unsafe_allow_html=True)

	add_selectbox = st.sidebar.selectbox(
		"Categorias",
		("Sobre", "Regressão")
	)

	match add_selectbox:

		case 'Sobre': about_card()

		case 'Regressão': regression_page()


