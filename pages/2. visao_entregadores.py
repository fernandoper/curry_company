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

st.set_page_config( page_title= 'Visão Entregadores', page_icon='📢', layout= 'wide')

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
    df.loc[:, 'Delivery_person_ID'] = df.loc[:,
                                             'Delivery_person_ID'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:,
                                               'Road_traffic_density'].str.strip()
    df.loc[:, 'Delivery_person_Age'] = df.loc[:,
                                              'Delivery_person_Age'].str.strip()
    df.loc[:, 'Delivery_person_Ratings'] = df.loc[:,
                                                  'Delivery_person_Ratings'].str.strip()
    df.loc[:, 'Time_Orderd'] = df.loc[:, 'Time_Orderd'].str.strip()
    df.loc[:, 'multiple_deliveries'] = df.loc[:,
                                              'multiple_deliveries'].str.strip()
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

# OVERALL METRICS (4 COLUMS)


def overall_metrics(df):
    col1, col2, col3, col4 = st.columns(4, gap='large')
    with col1:
        maior_idade = df.loc[:, 'Delivery_person_Age'].max()
        col1.metric('Max Age', maior_idade)
    with col2:
        menor_idade = df.loc[:, 'Delivery_person_Age'].min()
        col2.metric('Min Age', menor_idade)
    with col3:
        melhor_cond = df.loc[:, 'Vehicle_condition'].max()
        col3.metric('Best Veic Cond', melhor_cond)
    with col4:
        pior_cond = df.loc[:, 'Vehicle_condition'].min()
        col4.metric('Worst Veic Cond', pior_cond)

# AVERAGE FUNCTION


def avg_rating(df):
    rating = (df.loc[:, ['Delivery_person_ID', 'Delivery_person_Ratings']]
              .groupby('Delivery_person_ID')
              .mean().reset_index())
    rating.columns = ['Delivery_person_ID', 'Rating']
    return st.dataframe(rating)


def avg_traffic(df):
    traffic = (df.loc[:, ['Delivery_person_Ratings', 'Road_traffic_density']]
               .groupby('Road_traffic_density')
               .agg({'Delivery_person_Ratings': ["mean", "std"]}))
    traffic.columns = ['delivery_mean', 'delivery_std']
    traffic = traffic.reset_index()
    return st.dataframe(traffic)


def avg_weather(df):
    weather = (df.loc[:, ['Delivery_person_Ratings', 'Weatherconditions']].groupby('Weatherconditions')
               .agg({'Delivery_person_Ratings': ["mean", "std"]}))
    weather.columns = ['delivery_mean', 'delivery_std']
    weather = weather.reset_index()
    return st.dataframe(weather)

# ALTERADO O ASCENDING PARA 'top_asc' para poder utilizar o True e False na mesma função


def delivery_time(df, top_asc):
    df1 = (df.loc[:, ['Delivery_person_ID', 'City', 'Time_taken(min)']]
           .groupby(['City', 'Delivery_person_ID'])
           .min().sort_values(['City', 'Time_taken(min)'],
                              ascending=top_asc).reset_index())

    df_aux1 = df1.loc[df1['City'] == 'Metropolitian', :].head(10)
    df_aux2 = df1.loc[df1['City'] == 'Urban', :].head(10)
    df_aux3 = df1.loc[df1['City'] == 'Semi-Urban', :].head(10)

    df3 = pd.concat([df_aux1, df_aux2, df_aux3]).reset_index(drop=True)

    return df3


# ====================================================================
# IMPORT DATASET
# ====================================================================
df = pd.read_csv('dataset/train.csv')
# rodar o streamlit no cmd: python -m streamlit run visao_entregadores_module.py
# identar automaticamente: shift + alt + L

# ====================================================================
# DATA CLEANSING
# ====================================================================
df = clean_code(df)

# ====================================================================
# LAYOUT SIDEBAR
# ====================================================================

st.header('Marketplace - Visão Entregadores')

# SIDEBAR LOGO
image = Image.open('logo3.png')
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
st.tabs(['Visão Gerencial',])

# ====================================================================
# CONTAINER 1
# ====================================================================
with st.container():
    st.markdown(' ##### Overall Metrics')
    overall_metrics(df)
    st.markdown("""___""")

# ====================================================================
# CONTAINER 2
# ====================================================================
with st.container():
    st.title('Avaliações')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##### Avaliações Médias por Entregador')
        avg_rating(df)

    with col2:
        st.markdown('##### Avaliação Média por Trânsito')
        avg_traffic(df)

        st.markdown('##### Avaliação Média por Clima')
        avg_weather(df)

# ====================================================================
# CONTAINER 3
# ====================================================================
with st.container():
    st.markdown("""___""")
    st.title('Velocidade de Entrega')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('##### Top 10 Entregadores mais Rápidos')
        df3 = delivery_time(df, top_asc=True)
        st.dataframe(df3)

    with col2:
        st.markdown('##### Top 10 Entregadores mais Lentos')
        df3 = delivery_time(df, top_asc=False)
        st.dataframe(df3)
