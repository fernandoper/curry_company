# Libraries
from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go
import re
import datetime as dt
from PIL import Image

# Libs necessárias
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import folium_static

st.set_page_config( page_title= 'Visão Empresa', page_icon='📈', layout= 'wide')

# ====================================================================
# FUNÇÕES
# ====================================================================

def clean_code(df):
    """
    FUNÇÃO DE DATA CLEANSING:
    1. REMOÇÃO ESPAÇOS EM BRANCO
    2. EXCLUSÃO LINHAS NAN
    3. FORMATAÇÃO DAS COLUNAS INT, FLOAT, DATA E TIME

    INPUT: DATAFRAME
    OUTPUT: DATAFRAME
    """
    # Remove espaços em branco
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Delivery_person_ID'] = df.loc[:,'Delivery_person_ID'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:,'Road_traffic_density'].str.strip()
    df.loc[:, 'Delivery_person_Age'] = df.loc[:,'Delivery_person_Age'].str.strip()
    df.loc[:, 'Delivery_person_Ratings'] = df.loc[:,'Delivery_person_Ratings'].str.strip()
    df.loc[:, 'Time_Orderd'] = df.loc[:, 'Time_Orderd'].str.strip()
    df.loc[:, 'multiple_deliveries'] = df.loc[:,'multiple_deliveries'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()

    # Exclusão de linhas NaN
    linhas_vazias = df['Delivery_person_Age'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_vazias = df['Delivery_person_Ratings'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_vazias = df['Time_Orderd'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_vazias = df['Road_traffic_density'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_vazias = df['multiple_deliveries'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_vazias = df['Festival'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_vazias = df['City'] != 'NaN'
    df = df.loc[linhas_vazias, :]

    # Formatação colunas
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(int)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(int)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    df['Time_taken(min)'] = df['Time_taken(min)'].apply(
        lambda x: x.split('(min) ')[1])
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)

    return df

# CHART1: ORDERS BY DATE
def order_metric(df):
    df_aux = df.loc[:, ['ID', 'Order_Date']].groupby(
        'Order_Date').count().reset_index()
    df_aux.columns = ['Order_Date', 'Orders_Quantity']
    fig = px.bar(df_aux, x='Order_Date', y='Orders_Quantity')
    return fig

# CHART2: ORDERS BY TRAFFIC DENSITY TYPES
def traffic_order_share(df):
    df_aux = (df.loc[:, ['ID', 'Road_traffic_density']]
              .groupby('Road_traffic_density')
              .count()
              .reset_index())
    df_aux['Perc_ID'] = round(100 * (df_aux['ID'] / df_aux['ID'].sum()), 2)
    fig = px.pie(df_aux, values='Perc_ID', names='Road_traffic_density')
    return fig

# CHART3: ORDERS BY CITY AND TRAFFIC DENSITY
def traffic_order_city(df):
    df_aux = (df.loc[:, ['ID', 'City', 'Road_traffic_density']]
              .groupby(['City', 'Road_traffic_density'])
              .count()
              .reset_index())
    df_aux['perc_ID'] = round(100 * (df_aux['ID'] / df_aux['ID'].sum()), 2)
    fig = px.bar(df_aux, x='City', y='ID',
                 color='Road_traffic_density', barmode='group')
    return fig

# CHART4: ORDERS BY WEEK
def order_by_week(df):
    df['Week_of_Year'] = df['Order_Date'].dt.strftime("%U")
    df_aux = (df.loc[:, ['ID', 'Week_of_Year']]
                .groupby('Week_of_Year')
                .count()
                .reset_index())
    df_aux.columns = ['Week_of_Year', 'Qtde_Entregas']
    fig = px.bar(df_aux, x='Week_of_Year', y='Qtde_Entregas')
    return fig

# CHART5: ORDERS BY PERSON BY WEEK
def order_by_person_week(df):
    df['Week_of_Year'] = df['Order_Date'].dt.strftime("%U")
    df_aux1 = (df.loc[:, ['ID', 'Week_of_Year']]
               .groupby(['Week_of_Year'])
               .count()
               .reset_index())
    df_aux2 = (df.loc[:, ['Delivery_person_ID', 'Week_of_Year']]
               .groupby(['Week_of_Year'])
               .nunique()
               .reset_index())
    df_aux = pd.merge(df_aux1, df_aux2, how='inner')
    df_aux['Order_by_Delivery'] = round(df_aux['ID'] / df_aux['Delivery_person_ID'], 2)
    fig = px.line(df_aux, x='Week_of_Year', y='Order_by_Delivery')
    return fig

# CHART6: CITY AND TRAFFIC DENSITY LOCALIZATION
def country_maps (df):
    cols = ['City', 'Road_traffic_density','Delivery_location_latitude', 'Delivery_location_longitude']
    columns_groupby = ['City', 'Road_traffic_density']
    data_plot = df.loc[:, cols].groupby(columns_groupby).median().reset_index()
    map = folium.Map(zoom_start=15)
    for index, location_info in data_plot.iterrows():
        folium.Marker([location_info['Delivery_location_latitude'],
                    location_info['Delivery_location_longitude']],
                    popup=location_info[['City', 'Road_traffic_density']]).add_to(map)
    folium_static(map, width=1024, height=600)
    return None

# ====================================================================
# IMPORT DATASET
# ====================================================================
df = pd.read_csv('dataset/train.csv')
# rodar o streamlit no cmd: python -m streamlit run visao_empresa_module.py
# identar automaticamente: shift + alt + L

# ====================================================================
# DATA CLEANSING
# ====================================================================
df = clean_code(df)

# ====================================================================
# LAYOUT SIDEBAR
# ====================================================================
st.header('Marketplace - Visão Cliente')

# SIDEBAR LOGO
image = Image.open('logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('# Curry Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown(""" ___ """)
st.sidebar.markdown('## Selecione uma data limite')

# SIDEBAR FILTROS
date_slider = st.sidebar.slider(
    'Até qual valor?',
    value=dt.datetime(2022, 4, 6),
    min_value=dt.datetime(2022, 2, 11),
    max_value=dt.datetime(2022, 4, 6),
    format='DD-MM-YYYY')

st.sidebar.markdown(""" ___ """)

traffic_options = st.sidebar.multiselect(
    'Quais as condições do trânsito',
    ['Low', "Medium", 'High', 'Jam'],
    default=['Low', "Medium", 'High', 'Jam'])

st.sidebar.markdown(""" ___ """)
st.sidebar.markdown('### Powered by Comunidade DS')

# FILTRO DE DATA - A DATA VAI SER MENOR QUE A SELECIONADA NO SLIDER
linhas_selecionadas = df['Order_Date'] < date_slider
df = df.loc[linhas_selecionadas, :]

# FILTRO DE TRAFFIC
linhas_selecionadas = df['Road_traffic_density'].isin(traffic_options)
df = df.loc[linhas_selecionadas, :]

# ====================================================================
# LAYOUT TABS
# ====================================================================

tab1, tab2, tab3 = st.tabs(
    ['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

# ====================================================================
# VISÃO GERENCIAL
# ====================================================================

with tab1:
    with st.container():
        # CHART1: ORDERS BY DATE
        fig = order_metric(df)
        st.markdown('## Orders by Day')
        st.plotly_chart(fig, use_container_width=True)

    with st.container():

        col1, col2 = st.columns(2)
        with col1:
            # CHART2: ORDERS BY TRAFFIC DENSITY TYPES
            fig = traffic_order_share(df)
            st.markdown('## Orders by Traffic Density')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # CHART3: ORDERS BY CITY AND TRAFFIC DENSITY
            fig = traffic_order_city(df)
            st.markdown('## Orders by City and Traffic Density')
            st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# VISÃO TÁTICA
# ====================================================================

with tab2:
    with st.container():
        # CHART4: ORDERS BY WEEK
        fig = order_by_week(df)
        st.markdown('## Orders by Week')
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        # CHART5: ORDERS BY PERSON BY WEEK
        fig = order_by_person_week(df)
        st.markdown('## Orders by Person by Week')
        st.plotly_chart(fig, use_container_width=True)


# ====================================================================
# VISÃO GEOGRÁFICA
# ====================================================================

with tab3:
    with st.container():              
        # CHART6: CITY AND TRAFFIC DENSITY LOCALIZATION
        st.markdown('## City and Traffic Density Maps')        
        country_maps (df)
        

