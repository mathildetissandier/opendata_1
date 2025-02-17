import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Flask
from layout import layout_home, layout_logement, layout_sante, layout_predictions
# Assurez-vous d'importer la variable 'layout' définie dans layout_transports
from layout.layout_transports import layout

from app import app
from app import server

# Mise en page avec navigation
def serve_layout():
    return html.Div(
        style={'backgroundColor': 'black', 'color': 'white'},  # Fond noir, texte en blanc
        children=[
            dcc.Location(id='url', refresh=False),
            dbc.Container(id='page-content', fluid=True)
        ]
    )

app.layout = serve_layout
layout_sante.register_callbacks(app)
layout_predictions.register_callbacks(app)
# Callback pour gérer la navigation
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/transports':
        return layout  # Retourne la variable layout définie dans layout_transports.py
    elif pathname == '/logement':
        return layout_logement.layout
    elif pathname == '/sante':
        return layout_sante.layout
    elif pathname == '/predictions':
        return layout_predictions.layout
    else:
        return layout_home.layout

# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True)
