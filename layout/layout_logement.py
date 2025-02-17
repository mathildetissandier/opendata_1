import os
import pandas as pd
import dash
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import folium
import geopandas as gpd
import branca
from ipywidgets import interact
from IPython.display import display, clear_output
from pathlib import Path
from dash import callback_context, no_update

import pathlib
from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

# Chargement des données
average_price = pd.read_excel("data/Average price.xlsx")
sales_volume = pd.read_csv("data/Sales Volume.csv", delimiter=";")
vacants = pd.read_excel("data/vacants.xlsx")
affordability = pd.read_excel("data/affordability.xlsx") 
lutte = pd.read_excel("data/lutte_logements.xlsx") 

average_price2 = pd.read_excel("data/average_price2.xlsx")
sales_volume2 = pd.read_excel("data/sales_volume2.xlsx")
crimes2 = pd.read_excel("data/crimes2.xlsx")

# selection des années communes 
average_price2 = average_price2[(average_price2["Year"] >= 2010) & (average_price2["Year"] <= 2023)]
sales_volume2 = sales_volume2[(sales_volume2["Year"] >= 2010) & (sales_volume2["Year"] <= 2023)]
vacants2 = vacants[(vacants["Year"] >= 2010) & (vacants["Year"] <= 2023)]
crimes2 = crimes2[(crimes2["Year"] >= 2010) & (crimes2["Year"] <= 2023)]
affordability2 = affordability[(affordability["Year"] >= 2010) & (affordability["Year"] <= 2023)]

# Dictionnaire de traduction des mois
month_translation = {
    'janv.': 'Jan', 'févr.': 'Feb', 'mars': 'Mar', 'avr.': 'Apr', 'mai': 'May',
    'juin': 'Jun', 'juil.': 'Jul', 'août': 'Aug', 'sept.': 'Sep', 'oct.': 'Oct',
    'nov.': 'Nov', 'déc.': 'Dec'
}

# Transformation des dates
average_price['Time'] = average_price['Time'].replace(month_translation, regex=True)
average_price['Time'] = average_price['Time'].str.replace('-', ' ', regex=False)
average_price['Time'] = pd.to_datetime(average_price['Time'], format='%b %y')

sales_volume['Time'] = sales_volume['Time'].replace(month_translation, regex=True)
sales_volume['Time'] = sales_volume['Time'].str.replace('-', ' ', regex=False)
sales_volume['Time'] = pd.to_datetime(sales_volume['Time'], format='%b %y')

vacants['Year'] = pd.to_datetime(vacants['Year'], format='%Y')  # Assurez-vous que les années sont bien formatées
affordability['Year'] = pd.to_datetime(affordability['Year'], format='%Y')  # Assurez-vous que les années sont bien formatées

# Corrélation
scaler = MinMaxScaler()

sales_volume2_normalized = pd.DataFrame(scaler.fit_transform(sales_volume2.iloc[:, 1:]), columns=sales_volume2.columns[1:])
crimes2_normalized = pd.DataFrame(scaler.fit_transform(crimes2.iloc[:, 1:]), columns=crimes2.columns[1:])
average_price_N = pd.DataFrame(scaler.fit_transform(average_price.iloc[:, 1:]), columns=average_price.columns[1:])
vacants_N = pd.DataFrame(scaler.fit_transform(vacants.iloc[:, 1:]), columns=vacants.columns[1:])
average_price2_N = pd.DataFrame(scaler.fit_transform(average_price2.iloc[:, 1:]), columns=average_price2.columns[1:])

corr_sales_crimes = sales_volume2_normalized.corrwith(crimes2_normalized).dropna()
corr_vacants_price = vacants_N.corrwith(average_price_N).dropna()
corr_crimes_price = crimes2_normalized.corrwith(average_price2_N).dropna()

def create_corr_heatmap(corr_data, title):
    # Créer un tableau de corrélation avec Plotly
    fig = go.Figure(data=go.Heatmap(
        z=[corr_data.values],
        x=corr_data.index,  
        y=["Correlation"],
        colorscale="RdBu",  
        zmin=-1, zmax=1, 
        colorbar=dict(title="Corrélation"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Quartiers",
        yaxis_title="Corrélation",
        xaxis=dict(tickangle=45),
        template="plotly_dark"
    )
    
    return fig

# Traitement des données "lutte"
category_translation = {
    'Definitely will struggle to meet these payments': "Difficulté à payer (certain)",
    'Probably will struggle to meet these payments': "Difficulté à payer (probable)",
    'Probably will not struggle to meet these payments': "Facilité à payer (probable)",
    'Definitely will not struggle to meet these payments': "Facilité à payer (certain)",
    'Don’t know': "Ne sait pas"
}

tenure_translation = {
    'Home owner': "Propriétaire",
    'Private tenant': "Locataire privé",
    'Social tenant': "Locataire social"
}

# Convertir 'Month' en datetime avec le bon format
lutte["Month"] = pd.to_datetime(lutte["Month"], format="%d/%m/%Y")
lutte["Month_str"] = lutte["Month"].dt.strftime("%Y-%m")

# Appliquer la traduction
lutte['Category_fr'] = lutte['Category'].map(category_translation)
lutte['Tenure_fr'] = lutte['Tenure'].map(tenure_translation)

# Liste des mois disponibles, triés
months_sorted = sorted(lutte["Month"].dt.strftime("%Y-%m").unique())
months_indices = {month: i for i, month in enumerate(months_sorted)}

# Carte abrordabilité
# Charger le fichier GeoJSON des boroughs de Londres
gdf = gpd.read_file('data/london_boroughs.geojson')

# Fonction pour créer la carte selon l'année sélectionnée
def create_affordability_map(year):
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

    # Transformer le DataFrame en format long
    affordability = pd.read_excel("data/affordability.xlsx") 
    df_long = affordability.melt(id_vars=["Year"], var_name="name", value_name="affordability")

    # Filtrer les données pour l'année sélectionnée
    df_year = df_long[df_long["Year"] == year]

    # Fusionner avec gdf pour récupérer la géométrie
    gdf_merged = gdf.merge(df_year, on="name", how="left")

    # Créer une échelle de couleurs 
    colormap = branca.colormap.linear.Reds_09.scale(
        gdf_merged["affordability"].min(), 
        gdf_merged["affordability"].max()
    )

    # Ajouter chaque quartier sur la carte avec sa couleur
    for _, row in gdf_merged.iterrows():
        color = colormap(row["affordability"]) if pd.notna(row["affordability"]) else "grey"

        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": "black",
                "weight": 0.3,
                "fillOpacity": 0.9
            },
            tooltip=f"{row['name']} ({year}): {row['affordability']:.2f}",
        ).add_to(m)

    # Ajouter la légende
    colormap.caption = f"Abordabilité des quartiers en {year}"
    m.add_child(colormap)

    return m

def create_vacants_map(year):
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

    # Transformer le DataFrame en format long
    vacants = pd.read_excel("data/vacants.xlsx")
    # Transformer le DataFrame en format long
    df_long = vacants.melt(id_vars=["Year"], var_name="name", value_name="vacants")

    # Filtrer les données pour l'année sélectionnée
    df_year = df_long[df_long["Year"] == year]

    # Fusionner avec gdf pour récupérer la géométrie
    gdf_merged = gdf.merge(df_year, on="name", how="left")

    # Créer une échelle de couleurs 
    colormap = branca.colormap.linear.Blues_09.scale(
        gdf_merged["vacants"].min(), 
        gdf_merged["vacants"].max()
    )

    # Ajouter chaque quartier sur la carte avec sa couleur
    for _, row in gdf_merged.iterrows():
        color = colormap(row["vacants"]) if pd.notna(row["vacants"]) else "grey"

        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": "black",
                "weight": 0.3,
                "fillOpacity": 0.8
            },
            tooltip=f"{row['name']} ({year}): {row['vacants']:.2f}",
        ).add_to(m)

    # Ajouter la légende
    colormap.caption = f"Nombre de logements vacants {year}"
    m.add_child(colormap)

    return m

# Fonction pour rendre la carte en HTML
def render_map_html(year):
    m = create_affordability_map(year)
    return m._repr_html_()

def render_map_html_vacants(year):
    m = create_vacants_map(year)
    return m._repr_html_()

# Layout Dash
layout = dbc.Container([
    dbc.Button("⬅ Retour à l'accueil", href="/", color="primary", className="mb-3"),
    html.H2("Indicateurs sur le logements à Londres", className="text-center mb-4"),
 
    # 📌 Graphique de l'évolution des prix moyens
    html.Div([
        html.H3("Évolution du prix moyen d'un bien par quartier entre 1995 et 2024"),
        html.Label("Sélectionnez un quartier :"),
        dcc.Dropdown(
            id='quartier-dropdown-logement',
            options=[{'label': "Tous les quartiers", 'value': "all"}] +
                    [{'label': col, 'value': col} for col in average_price.columns[1:]],
            value="all",
            style={'width': '50%', 'color': 'black'}
        ),
        dcc.Graph(id='logement-graph'),
        dbc.Button("Analyse", id="open-analyses-button-1", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin-bottom': '20px'}),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses de l'Évolution du prix moyen d'un bien par quartier entre 1995 et 2024")),
            dbc.ModalBody(id="analyses-content-1"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-1", className="ms-auto")),
        ], id="analyses-modal-1", size="lg", is_open=False),

    # 📌 Graphique de l'évolution des ventes de biens
    html.Div([
        html.H3("Évolution du nombre de biens vendus par quartier entre 1995 et 2024"),
        html.Label("Sélectionnez un quartier :"),
        dcc.Dropdown(
            id='quartier-dropdown-ventes',
            options=[{'label': "Tous les quartiers", 'value': "all"}] +
                    [{'label': col, 'value': col} for col in sales_volume.columns[1:]],
            value="all",
            style={'width': '50%', 'color': 'black'}
        ),
        dcc.Graph(id='ventes-graph'),
        dbc.Button("Analyse", id="open-analyses-button-2", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin-bottom': '20px'}),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses de l'Évolution du nombre de biens vendus par quartier entre 1995 et 2024")),
            dbc.ModalBody(id="analyses-content-2"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-2", className="ms-auto")),
        ], id="analyses-modal-2", size="lg", is_open=False),
   
    # 📌 Graphique de l'évolution de l'abordabilité
    html.Div([
        # 📌 Colonne 1 : Graphique de l'évolution de l'abordabilité
        html.Div([
            html.H3("Évolution du ratio prix/revenus (abordabilité) entre 1997 et 2023"),
            html.Label("Sélectionnez un quartier :"),
            dcc.Dropdown(
                id='quartier-dropdown-affordability',
                options=[{'label': "Tous les quartiers", 'value': "all"}] +
                        [{'label': col, 'value': col} for col in affordability.columns[1:]],
                value="all",
                style={'width': '100%', 'color': 'black'}
            ),
            dcc.Graph(id='affordability-graph'),
            dbc.Button("Analyse", id="open-analyses-button-3", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
        ], style={'width': '50%', 'padding': '10px'}),

        # 📌 Colonne 2 : Carte interactive de l'abordabilité
        html.Div([
            html.H3("Carte de l'abordabilité des quartiers à Londres"),
            html.Label("Sélectionnez une année :"),
            dcc.Slider(
                id='year-slider',
                min=1997,
                max=2023,
                value=2013,
                marks={i: str(i) for i in range(1997, 2024,2)},
                step=1
            ),
            html.Iframe(
                id="affordability-map",
                srcDoc=render_map_html(2013),
                width="100%",
                height="500px",
                style={"border": "none"}
            )
        ], style={'width': '50%', 'padding': '10px'}),
    ], style={'padding': '10px','display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'border': '1px solid #ccc','margin-bottom': '20px'}),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses de l'abordabilité des quartiers à Londres")),
            dbc.ModalBody(id="analyses-content-3"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-3", className="ms-auto")),
        ], id="analyses-modal-3", size="lg", is_open=False),

    # 📌 Graphique de l'évolution des logements vacants
        html.Div([
            # 📌 Colonne 1 : carte de l'évolution des logements vacants
                html.Div([
                html.H3("Carte des logements vacants à Londres"),
                html.Label("Sélectionnez une année :"),
                dcc.Slider(
                    id='year-slider-2',
                    min=2004,
                    max=2023,
                    value=2013,
                    marks={i: str(i) for i in range(2004, 2024,2)},
                    step=1
                ),
                html.Iframe(
                    id="vacants-map",
                    srcDoc=render_map_html_vacants(2013),
                    width="100%",
                    height="500px",
                    style={"border": "none"}
                ),dbc.Button("Analyse", id="open-analyses-button-4", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
            ], style={'width': '50%', 'padding': '10px'}),

        # 📌 Colonne 2 : evolution des logements vacants
        html.Div([
            html.H3("Évolution des logements vacants par quartier entre 2004 et 2023"),
            html.Label("Sélectionnez un quartier :"),
            dcc.Dropdown(
                id='quartier-dropdown-vacants',
                options=[{'label': "Tous les quartiers", 'value': "all"}] +
                        [{'label': col, 'value': col} for col in vacants.columns[1:]],
                value="all",
                style={'width': '50%', 'color': 'black'}
            ),
            dcc.Graph(id='vacants-graph'),
        ], style={'padding': '10px', 'margin-bottom': '20px'}),
    ], style={'padding': '10px','display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'border': '1px solid #ccc','margin-bottom': '20px'}),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses des logements vacants par quartiers à Londres")),
            dbc.ModalBody(id="analyses-content-4"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-4", className="ms-auto")),
        ], id="analyses-modal-4", size="lg", is_open=False),

    # 📌 Graphique de l'évolution des difficultés de paiement (sunburst)
    html.Div([
    html.H3("Pression financière du logement à Londres"),
    dcc.Slider(
        id='month-slider',
        min=0,
        max=len(months_sorted) - 1,
        value=0,  
        marks={i: month for i, month in enumerate(months_sorted)}, 
        step=1
    ),
    html.Div(
        dcc.Graph(id='sunburst-graph'),
        style={
            'display': 'flex',         
            'justifyContent': 'center', 
            'alignItems': 'center',    
            'width': '100%'            
        }
    ),dbc.Button("Analyse", id="open-analyses-button-5", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
    ], style={
        'padding': '10px', 
        'border': '1px solid #ccc', 
        'margin-bottom': '20px'
    }),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses de la Pression financière du logement à Londres")),
            dbc.ModalBody(id="analyses-content-5"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-5", className="ms-auto")),
        ], id="analyses-modal-5", size="lg", is_open=False),

    # 📌 Matrice de corrélation
    html.Div([
        # Premier graphique
        html.Div([
            html.H3("Matrice de corrélation entre quartiers"),
            dcc.RadioItems(
                id='corr-radio',
                options=[
                    {'label': 'Prix Moyen', 'value': 'Prix Moyen'},
                    {'label': 'Nombre de ventes', 'value': 'Nombre de biens vendus'},
                    {'label': 'Nombre de logements vacants', 'value': 'Nombre de logements vacants'},
                ],
                value='Nombre de logements vacants',  # Valeur par défaut
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            ),
            dcc.Graph(id='corr-matrix-graph'),
            dbc.Button("Analyse", id="open-analyses-button-6", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin-bottom': '20px', 'flex': '1'}),
            dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses de la corrélation entre quartiers")),
            dbc.ModalBody(id="analyses-content-6"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-6", className="ms-auto")),
        ], id="analyses-modal-6", size="lg", is_open=False),

        # Deuxième graphique
        html.Div([
            html.H3("Matrice de corrélation par quartiers"),
            dcc.RadioItems(
                id='corr-radio-2',
                options=[
                    {'label': 'Nombre de ventes & Nombre de crimes', 'value': 'Sales & Crimes'},
                    {'label': 'Nombre de logements vacants & Prix moyen', 'value': 'Vacants & Price'},
                    {'label': 'Nombre de crimes & Prix moyen', 'value': 'Crimes & Price'},
                ],
                value='Vacants & Price',  # Valeur par défaut
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            ),
            dcc.Graph(id='corr-matrix-graph-2',responsive=True,),
            dbc.Button("Analyse", id="open-analyses-button-7", color="info", className="mb-3", style={'font-size': '18px', 'padding': '15px 30px'}),
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'margin-bottom': '20px', 'flex': '1'}),
                    dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("analyses de la corrélation par quartiers")),
            dbc.ModalBody(id="analyses-content-7"),  # Contenu dynamique chargé ici
            dbc.ModalFooter(dbc.Button("Fermer", id="close-analyses-button-7", className="ms-auto")),
        ], id="analyses-modal-7", size="lg", is_open=False)
    ], style={'display': 'flex', 'gap': '20px'}),
], fluid=True)

# Callback pour mettre à jour le graphique des prix
@dash.callback(
    Output('logement-graph', 'figure'),
    Input('quartier-dropdown-logement', 'value')
)
def update_price_graph(selected_quartier):
    fig = make_subplots(rows=1, cols=1)

    if selected_quartier == "all":
        for column in average_price.columns[1:]:
            fig.add_trace(go.Scatter(x=average_price['Time'], y=average_price[column], mode='lines', name=column))
    else:
        fig.add_trace(go.Scatter(x=average_price['Time'], y=average_price[selected_quartier], mode='lines', name=selected_quartier))

    fig.update_layout(
        title=f"Évolution du prix moyen d'un bien {('par quartier' if selected_quartier == 'all' else f'pour {selected_quartier}')}",
        xaxis_title="Temps",
        yaxis_title="Prix",
        template="plotly_dark"
    )
    return fig

# Callback pour mettre à jour le graphique des ventes
@dash.callback(
    Output('ventes-graph', 'figure'),
    Input('quartier-dropdown-ventes', 'value')
)
def update_sales_graph(selected_quartier):
    fig = make_subplots(rows=1, cols=1)

    if selected_quartier == "all":
        for column in sales_volume.columns[1:]:
            fig.add_trace(go.Scatter(x=sales_volume['Time'], y=sales_volume[column], mode='lines', name=column))
    else:
        fig.add_trace(go.Scatter(x=sales_volume['Time'], y=sales_volume[selected_quartier], mode='lines', name=selected_quartier))

    fig.update_layout(
        title=f"Évolution du nombre de biens vendus {('par quartier' if selected_quartier == 'all' else f'pour {selected_quartier}')}",
        xaxis_title="Temps",
        yaxis_title="Nombre de ventes",
        template="plotly_dark"
    )
    return fig

# Callback pour mettre à jour le graphique des logements vacants
@dash.callback(
    Output('vacants-graph', 'figure'),
    Input('quartier-dropdown-vacants', 'value')
)
def update_vacants_graph(selected_quartier):
    fig = go.Figure()

    if selected_quartier == "all":
        for area in vacants.columns[1:]:
            fig.add_trace(go.Scatter(
                x=vacants['Year'], 
                y=vacants[area], 
                mode='lines+markers', 
                name=area
            ))
        fig.update_layout(
            title="Évolution des logements vacants par quartier entre 2004 et 2023",
            xaxis_title="Année",
            yaxis_title="Nombre de logements vacants",
            template="plotly_dark"
        )
    else:
        fig.add_trace(go.Scatter(
            x=vacants['Year'], 
            y=vacants[selected_quartier], 
            mode='lines+markers', 
            name=selected_quartier
        ))
        fig.update_layout(
            title=f"Évolution des logements vacants entre 2004 et 2023 - {selected_quartier}",
            xaxis_title="Année",
            yaxis_title="Nombre de logements vacants",
            template="plotly_dark"
        )

    return fig

# Callback pour mettre à jour le graphique de l'abordabilité
@dash.callback(
    Output('affordability-graph', 'figure'),
    Input('quartier-dropdown-affordability', 'value')
)
def update_affordability_graph(selected_quartier):
    fig = go.Figure()

    if selected_quartier == "all":
        for area in affordability.columns[1:]:
            fig.add_trace(go.Scatter(
                x=affordability['Year'], 
                y=affordability[area], 
                mode='lines+markers', 
                name=area
            ))
        fig.update_layout(
            title="Évolution du ratio prix/revenus (abordabilité) entre 1997 et 2023",
            xaxis_title="Année",
            yaxis_title="Ratio prix/revenus",
            template="plotly_dark"
        )
    else:
        fig.add_trace(go.Scatter(
            x=affordability['Year'], 
            y=affordability[selected_quartier], 
            mode='lines+markers', 
            name=selected_quartier
        ))
        fig.update_layout(
            title=f"Évolution du ratio prix/revenus (abordabilité) entre 1997 et 2023 - {selected_quartier}",
            xaxis_title="Année",
            yaxis_title="Ratio prix/revenus",
            template="plotly_dark"
        )

    return fig

# Callback pour mettre à jour la matrice de corrélation
@dash.callback(
    Output('corr-matrix-graph', 'figure'),
    Input('corr-radio', 'value')
)
def update_corr_matrix(selected):
    fig = go.Figure()

    if selected == 'Prix Moyen':
        corr_matrix = average_price.iloc[:, 1:].corr()
    elif selected == 'Nombre de biens vendus':
        corr_matrix = sales_volume.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').corr()
    elif selected == 'Nombre de logements vacants':
        corr_matrix = vacants.iloc[:, 1:].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        hoverongaps=False
    ))

    fig.update_layout(
        title=f"Corrélation entre quartiers selon : {selected}",
        xaxis_title="Quartiers",
        yaxis_title="Quartiers",
        xaxis=dict(tickmode='array', tickvals=np.arange(len(corr_matrix.columns)), ticktext=corr_matrix.columns),
        yaxis=dict(tickmode='array', tickvals=np.arange(len(corr_matrix.columns)), ticktext=corr_matrix.columns),
        template="plotly_dark",
        height=800,  
        width=800,  
    )

    return fig

# Callback pour mettre à jour le graphique de corrélation en fonction de la sélection
@dash.callback(
    Output('corr-matrix-graph-2', 'figure'),
    Input('corr-radio-2', 'value')
)
def update_corr_graph(selected_corr):
    if selected_corr == 'Sales & Crimes':
        corr_data = corr_sales_crimes
        title = "Corrélation par quartiers entre : Nombre de ventes & Nombre de crimes"
    elif selected_corr == 'Vacants & Price':
        corr_data = corr_vacants_price
        title = "Corrélation par quartiers entre : Nombre de logements vacants & Prix moyen"
    elif selected_corr == 'Crimes & Price':
        corr_data = corr_crimes_price
        title = "Corrélation par quartiers entre : Nombre de crimes vacants & Prix moyen"

    return create_corr_heatmap(corr_data, title)

# Callback pour mettre à jour le graphique Sunburst selon le mois sélectionné
@dash.callback(
    Output('sunburst-graph', 'figure'),
    Input('month-slider', 'value')
)
def update_sunburst(selected_index):
    selected_month = months_sorted[selected_index]
    filtered_df = lutte[lutte["Month_str"] == selected_month]

    fig = px.sunburst(
        filtered_df, 
        path=['Category_fr', 'Tenure_fr'], 
        values='Value',
        color='Category_fr',
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_traces(textinfo="label+percent parent", insidetextorientation="radial")

    fig.update_layout(
        width=800, height=800,
        paper_bgcolor="black",  # Fond noir
        plot_bgcolor="black",   # Fond noir aussi pour le graph
        font=dict(size=14, color="white"),  # Texte blanc
        title_font=dict(size=20, color="white"),  # Titre blanc
    )
    return fig

# Callback Dash pour mettre à jour la carte en fonction de l'année
@dash.callback(
    Output('affordability-map', 'srcDoc'),
    [Input('year-slider', 'value')]
)
def update_map(year):
    return render_map_html(year)

# Callback Dash pour mettre à jour la carte en fonction de l'année
@dash.callback(
    Output('vacants-map', 'srcDoc'),
    [Input('year-slider-2', 'value')]
)
def update_map2(year):
    return render_map_html_vacants(year)

def load_analyses(graphique_id):
    """
    Charge le contenu d'analyses pour un graphique donné.
    """
    chemin_fichier = Path(f"analyse/logement/analyse_graphique_{graphique_id}.html")
    try:
        with open(chemin_fichier, "r", encoding="utf-8") as fichier:
            return dcc.Markdown(
                fichier.read(),
                dangerously_allow_html=True  # Autorise l'utilisation de HTML
            )
    except FileNotFoundError:
        return dcc.Markdown("Aucune analyses disponible pour ce graphique.")

for i in range(1, 8):  # 10 graphiques => 10 modals
    @callback(
        [Output(f"analyses-modal-{i}", "is_open"),
         Output(f"analyses-content-{i}", "children")],
        [Input(f"open-analyses-button-{i}", "n_clicks"),
         Input(f"close-analyses-button-{i}", "n_clicks")],
        prevent_initial_call=True
    )
    def toggle_modal(open_clicks, close_clicks, i=i):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == f"open-analyses-button-{i}":
            # Charger l'analyses spécifique pour ce graphique
            analyses_text = load_analyses(i)  # Utiliser la fonction pour charger le fichier
            return True, analyses_text
        elif button_id == f"close-analyses-button-{i}":
            return False, no_update
        return no_update, no_update

 