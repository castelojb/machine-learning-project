import streamlit as st


class SkillsCard:
	project_skills = """
			* Gestão de Orçamentos
			* Estimativas de custos
			* Gerenciamento de Projetos
			* Projetista Eletrico
			"""

	program_skills = """
			* AutoCad
			* Revit
			* Sketchup
			* Word
			* Excel
			"""

	soft_skills = """
				* Trabalho em Equipe
				* Criatividade
				* Comunicativa
				* Organizada
				"""

	def __init__(self):
		pass
	def __call__(self):
		col_project_skills, col_software_skills, col_soft_skills = st.columns(3)

		with col_project_skills:
			st.header(':hammer: Projetos')

			st.markdown(self.project_skills)

		with col_software_skills:
			st.header(':computer: Programas')

			st.markdown(self.program_skills)

		with col_soft_skills:

			st.header(':coffee: Pessoais')

			st.markdown(self.soft_skills)






