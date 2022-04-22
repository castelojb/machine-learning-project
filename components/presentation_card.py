import streamlit as st


class PresentationCard:

	description = """Voltado a resultados, com experiência em trabalho com publico, e em atividades que requeiram tempo . Especialista em organização de dados com uso de software e, com experiência em uso e manutenção de dados. Muito detalhista e habilidoso na criação de planos de projetos detalhados e precisos."""

	def __init__(self, path_to_image: str, path_to_cv_file: str):
		self.path_to_image = path_to_image
		self.path_to_cv_file = path_to_cv_file

	def __call__(self):
		col_image_card, col_julianne_description = st.columns(2)

		with col_image_card:
			st.image(self.path_to_image)

		with col_julianne_description:

			st.header('OLÁ! ME CHAMO JULIANE GOMES DE OLIVEIRA')
			st.subheader('Estudante de engenharia civil e entusiasta da tecnologia')

			st.markdown(self.description)






