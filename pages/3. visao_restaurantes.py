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

st.set_page_config( page_title= 'Visão Restaurantes', page_icon='🍕', layout= 'wide')

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

# COL2 - AVG DISTANCE


def distance(df):
    cols = ['Restaurant_latitude', 'Restaurant_longitude',
            'Delivery_location_latitude', 'Delivery_location_longitude']
    df['Distance'] = df.loc[:, cols].apply(lambda x: haversine(
        (x['Restaurant_latitude'], x['Restaurant_longitude']),
        (x['Delivery_location_latitude'], x['Delivery_location_longitude'])), axis=1)
    avg_distance = round(df['Distance'].mean(), 2)
    return avg_distance

# COL3 a COL6 - AVG TIME AND STD


def avg_std_time_delivery(df, festival, op):
    """Essa função calcula o tempo médio e desvio padrão do tempo de entrega
    Args:
        df: dataframe com os dados necessários para o cálculo
        op: tipo de operação que precisa ser calculado

        Input:
            'avg_time': calcula tempo médio
            'std_time': calcula desvio padrão do tempo

        output: dataframe com duas colunas e uma linha
    """
    df_aux = df.loc[:, ['Festival', 'Time_taken(min)']].groupby(
        'Festival').agg({'Time_taken(min)': ['mean', 'std']})
    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()
    df_aux = np.round(df_aux.loc[df_aux['Festival'] == festival, op], 2)
    return df_aux

# CHART 1


def avg_time_delivery_by_city(df):
    df_aux = df.loc[:, ['Time_taken(min)', 'City']].groupby(
        'City').agg({'Time_taken(min)': ['mean', 'std']})
    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Control', x=df_aux['City'], y=df_aux['avg_time'], error_y=dict(
        type='data', array=df_aux['std_time'])))
    fig.update_layout(barmode='group')
    return fig

# CHART 2


def avg_time_by_city(df):
    cols = ['Restaurant_latitude', 'Restaurant_longitude',
            'Delivery_location_latitude', 'Delivery_location_longitude']
    df['Distance'] = df.loc[:, cols].apply(lambda x: haversine(
        (x['Restaurant_latitude'], x['Restaurant_longitude']),
        (x['Delivery_location_latitude'], x['Delivery_location_longitude'])), axis=1)
    avg_distance = df.loc[:, ['City', 'Distance']
                          ].groupby('City').mean().reset_index()
    fig = go.Figure(data=[go.Pie(labels=avg_distance['City'],
                    values=avg_distance['Distance'], pull=[0, 0, 0.06])])
    return fig

# CHART 3


def avg_std_by_city_traffic(df):
    df_aux = df.loc[:, ['Time_taken(min)', 'City', 'Road_traffic_density']].groupby(
        ['City', 'Road_traffic_density']).agg({'Time_taken(min)': ['mean', 'std']})
    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()
    fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'], values='avg_time', color='std_time',
                      color_continuous_scale='RdBu', color_continuous_midpoint=np.average(df_aux['std_time']))
    return fig

# CHART 4


def dist_distribution(df):
    df_aux = df.loc[:, ['Time_taken(min)', 'City', 'Type_of_order']].groupby(
        ['City', 'Type_of_order']).agg({'Time_taken(min)': ['mean', 'std']})
    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()
    return df_aux


# ====================================================================
# IMPORT DATASET
# ====================================================================
df = pd.read_csv('dataset/train.csv')
# rodar o streamlit no cmd: python -m streamlit run visao_restaurantes_module.py
# identar automaticamente: shift + alt + L

# ====================================================================
# DATA CLEANSING
# ====================================================================
df = clean_code(df)

# ====================================================================
# LAYOUT SIDEBAR
# ====================================================================

st.header('Marketplace - Visão Restaurantes')

# SIDEBAR LOGO
image = Image.open('logo2.png')
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

st.tabs(['Visão Gerencial'])

# ====================================================================
# VISÃO GERENCIAL
# ====================================================================

# ====================================================================
# CONTAINER 1 - OVERALL METRICS (6)
# ====================================================================
with st.container():
    st.markdown('## Overall Metrics')
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
            delivery_unique = df.loc[:, 'Delivery_person_ID'].nunique()
            col1.metric('Uniques ID', delivery_unique)

    with col2:
            avg_distance = distance(df)
            col2.metric('Avg Distance Deliveries', avg_distance)

    with col3:
            df_aux = avg_std_time_delivery(df, "Yes", 'avg_time')
            col3.metric('Avg Time Delivery w/ Fest', df_aux)

    with col4:
            df_aux = avg_std_time_delivery(df, "Yes", 'std_time')
            col4.metric('Avg Time Delivery w/out Fest', df_aux)

    with col5:
            df_aux = avg_std_time_delivery(df, "No", 'avg_time')
            col5.metric('Avg Time Delivery w/out Fest', df_aux)

    with col6:
            df_aux = avg_std_time_delivery(df, "No", 'std_time')
            col6.metric('Avg Time Delivery w/ Fest', df_aux)

# ====================================================================
# CONTAINER 2 - AVG TIME DELIVERY BY CITY
# ====================================================================
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('##### Tempo Médio de Entrega por Cidade')
            fig = avg_time_delivery_by_city(df)
            st.plotly_chart(fig)
            
        with col2:
            st.markdown('##### Distance Distribution')
            df_aux = dist_distribution(df)
            st.dataframe(df_aux)

# ====================================================================
# CONTAINER 3
# ====================================================================
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('##### Average Delivery Time by City')
            fig = avg_time_by_city(df)
            st.plotly_chart(fig)
            
        with col2:
            st.markdown('##### Avg e STD by City and Traffic')
            fig = avg_std_by_city_traffic(df)
            st.plotly_chart(fig)
