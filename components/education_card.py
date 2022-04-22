import streamlit as st


class EducationCard:


	def __init__(self):
		pass
	def __call__(self):

		col_first, col_second, col_third = st.columns(3)

		with col_first:
			st.header('2014 - 2016')

			st.markdown("Conclui meu ensino profissionalizante na Escola Estadual de Educação Profissional Maria Cavalcante Costa")

		with col_second:
			st.header('2016 - 2018')

			st.markdown("Comecei meus estudos em Engenharia Civil no Instituto Educacional e Tecnológico (CISNE)")

		with col_third:
			st.header('2018 - Atual')

			st.markdown("Decidi concluir meu curso no Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), onde estudo até hoje")



