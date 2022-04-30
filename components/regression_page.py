import pandas as pd
import streamlit as st

from components.text_content import regression_ml_introduction, regression_gd_explain, regression_gde_explain, \
	regression_ols_explain, regression_polinomial_explain
from ml_core.regression import GradientDescent, StochasticGradientDescent, OrdinaryLeastSquares
from ml_core.utils import Normalization, DataProcess

import plotly.express as px
import plotly.graph_objs as go

class RegressionPage:

	introduction_text = regression_ml_introduction

	gd_text = regression_gd_explain

	gde_text = regression_gde_explain

	ols_text = regression_ols_explain

	polinomial_text = regression_polinomial_explain

	def __init__(self):

		self.sample_df = pd.read_csv('data/artificial1d.csv', header=None, names=['x', 'y'])

		self.normalized_x, _ = Normalization.z_score_normalization(self.sample_df['x'].to_numpy())

		self.normalized_y, self.denormalized_y = Normalization.z_score_normalization(self.sample_df['y'].to_numpy())

	@staticmethod
	def __show_rmse_curve(history, title='RMSE Curve'):
		rmse_values = [step['rmse_error'] for step in history]

		steps = [i for i, _ in enumerate(history)]

		fig = px.line(x=steps, y=rmse_values, labels={'x': 'steps', 'y': 'RMSE'}, title=title)

		return fig

	@staticmethod
	def __show_final_result(history, denormalized_function, x, y, test_matrix, title='Model Result'):

		if isinstance(history, list):
			final_model = history[-1]['model']
		else:
			final_model = history

		preds = denormalized_function(final_model.predict(test_matrix))

		fig = px.scatter(x=x, y=y, title=title)

		fig.add_trace(
			go.Scatter(x=x, y=preds[:, 0])
		)

		return fig

	def __call__(self):

		X_ones = DataProcess.add_ones_column(self.normalized_x)
		y = DataProcess.reshape_vector(self.normalized_y)

		st.markdown(self.introduction_text)

		st.table(self.sample_df.head())

		st.markdown("<h1 style='text-align: center;'>Gradiente Descendente</h1>", unsafe_allow_html=True)
		st.markdown(self.gd_text)

		gd = GradientDescent(ephocs=500, with_history_predictions=True, l2_regulazation=0.001)
		history_gd = gd.fit(X_ones, y)

		_, plot_rmse_gd, plot_result_gd, _ = st.columns([0.1, 0.4, 0.4, 0.1])

		with plot_rmse_gd:
			st.plotly_chart(self.__show_rmse_curve(history_gd), use_container_width=True)

		with plot_result_gd:
			st.plotly_chart(self.__show_final_result(
				history_gd,
				self.denormalized_y,
				self.sample_df['x'],
				self.sample_df['y'],
				X_ones
			), use_container_width=True)

		st.markdown("<h1 style='text-align: center;'>Gradiente Descendente Estocastico</h1>", unsafe_allow_html=True)
		st.markdown(self.gde_text)

		gde = StochasticGradientDescent(ephocs=100, with_history_predictions=True, l2_regulazation=0.001)

		history_gde = gde.fit(X_ones, y)
		_, plot_rmse_gde, plot_result_gde, _ = st.columns([0.1, 0.4, 0.4, 0.1])

		with plot_rmse_gde:
			st.plotly_chart(self.__show_rmse_curve(history_gde), use_container_width=True)

		with plot_result_gde:
			st.plotly_chart(self.__show_final_result(
				history_gde,
				self.denormalized_y,
				self.sample_df['x'],
				self.sample_df['y'],
				X_ones
			), use_container_width=True)

		st.markdown("<h1 style='text-align: center;'>Mínimos Quadrados Ordinarios</h1>", unsafe_allow_html=True)
		st.markdown(self.ols_text)

		ols = OrdinaryLeastSquares()
		model_ols = ols.fit(X_ones, y)

		_, plot_ols, _ = st.columns([0.1, 0.8, 0.1])

		with plot_ols:
			st.plotly_chart(self.__show_final_result(
				model_ols,
				self.denormalized_y,
				self.sample_df['x'],
				self.sample_df['y'],
				X_ones
			), use_container_width=True)

		st.markdown("<h1 style='text-align: center;'>Regressão Polinomial</h1>", unsafe_allow_html=True)
		st.markdown(self.polinomial_text)

		X_pow = DataProcess.generate_polynomial_order(X_ones, 11)
		model_p_ols = ols.fit(X_pow, y)

		_, plot_p_ols, _ = st.columns([0.1, 0.8, 0.1])

		with plot_p_ols:
			st.plotly_chart(self.__show_final_result(
				model_p_ols,
				self.denormalized_y,
				self.sample_df['x'],
				self.sample_df['y'],
				X_pow
			), use_container_width=True)
