import dash
from dash import html, dcc, Input, Output
import pickle
import pandas as pd
import numpy as np
# -*- coding: utf-8 -*-


# Load the trained model and car data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned_car.csv')

# Create Dash application instance
app = dash.Dash(__name__)

# Define layout of the application with simplified styling
app.layout = html.Div(style={'backgroundColor': '#f2f2f2', 'padding': '20px', 'fontSize': '1.2rem'}, children=[
    html.Div(style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}, children=[
        html.H1("Car Price Predictor", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("Predict the price of a car based on its details", style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div(style={'margin': '0 auto', 'maxWidth': '500px'}, children=[
            html.Label("Select Company: "),
            dcc.Dropdown(
                id='company-dropdown',
                options=[{'label': company, 'value': company} for company in sorted(car['company'].unique())],
                value=None,
                clearable=False,
                className="form-control mb-3",
                style={'backgroundColor': '#f2f2f2', 'borderRadius': '5px', 'marginBottom': '10px'}
            ),
            html.Label("Select Model: "),
            dcc.Dropdown(id='model-dropdown', className="form-control mb-3", style={'backgroundColor': '#f2f2f2', 'borderRadius': '5px', 'marginBottom': '10px'}),
            html.Label("Select Year: "),
            dcc.Dropdown(id='year-dropdown', className="form-control mb-3", style={'backgroundColor': '#f2f2f2', 'borderRadius': '5px', 'marginBottom': '10px'}),
            html.Label("Select Fuel Type: "),
            dcc.Dropdown(id='fuel-type-dropdown', className="form-control mb-3", style={'backgroundColor': '#f2f2f2', 'borderRadius': '5px', 'marginBottom': '10px'}),
            html.Label("Enter Number of Kilometers Travelled: "),
            dcc.Input(id='kilometers-input', type='number', value=0, className="form-control mb-3", style={'backgroundColor': '#f2f2f2', 'borderRadius': '5px', 'marginBottom': '10px'}),
            html.Button('Predict Price', id='predict-button', n_clicks=0, style={'marginTop': '20px', 'width': '100%', 'backgroundColor': 'blue', 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px'})
        ])
    ]),
    html.Div(id='prediction-output', style={'textAlign': 'center'})
])

# Define callback to update car model options based on selected company
@app.callback(
    Output('model-dropdown', 'options'),
    [Input('company-dropdown', 'value')]
)
def update_model_options(selected_company):
    if selected_company:
        car_models = sorted(car[car['company'] == selected_company]['name'].unique())
        return [{'label': model, 'value': model} for model in car_models]
    else:
        return []

# Define callback to update year options based on selected company and model
@app.callback(
    Output('year-dropdown', 'options'),
    [Input('company-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_year_options(selected_company, selected_model):
    if selected_company and selected_model:
        years = sorted(car[(car['company'] == selected_company) & (car['name'] == selected_model)]['year'].unique(), reverse=True)
        return [{'label': str(year), 'value': year} for year in years]
    else:
        return []

# Define callback to update fuel type options based on selected company, model, and year
@app.callback(
    Output('fuel-type-dropdown', 'options'),
    [Input('company-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_fuel_type_options(selected_company, selected_model, selected_year):
    if selected_company and selected_model and selected_year:
        fuel_types = car[(car['company'] == selected_company) & (car['name'] == selected_model) & (car['year'] == selected_year)]['fuel_type'].unique()
        return [{'label': fuel_type, 'value': fuel_type} for fuel_type in fuel_types]
    else:
        return []

# Define callback to predict price
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('company-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('fuel-type-dropdown', 'value'),
     Input('kilometers-input', 'value')]
)
def predict_price(n_clicks, selected_company, selected_model, selected_year, selected_fuel_type, kilometers):
    if n_clicks > 0:
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                  data=np.array([selected_model, selected_company, selected_year, kilometers, selected_fuel_type]).reshape(1, 5)))
        return html.H4(f"Predicted Price: ₹{np.round(prediction[0], 2)}", style={'marginTop': '20px'})
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)
