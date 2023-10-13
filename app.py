import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

st.set_page_config(page_title="Precio Dólar Real", page_icon="📈")

st.title("Precio Dólar Real")

@st.cache_data(ttl=600)
def get_data():
    from streamlit_gsheets import GSheetsConnection
    conn = st.experimental_connection("gsheets", type=GSheetsConnection)
    df = conn.read(index_col=0, parse_dates=True)
    try:
        r = requests.get('https://dolarapi.com/v1/dolares/blue')
        dolar_blue_hoy = eval(r.text)
        df.loc[df.index[-1], ['venta_informal', 'informal_ajustado']] = dolar_blue_hoy['venta']
        return df
    except Exception as e:
        dolar_blue_hoy
        st.error('No se pudo acceder al valor del día de la fecha.', e)

df = get_data()

import locale
locale.setlocale(locale.LC_ALL,'es_ES.UTF-8')
import calendar
def aumento_porcentaje(x, y, puntos_porcentuales=False):
    if not puntos_porcentuales:
        return str(round((x/y-1)*100, 1))+'%'
    return str(round((x - y)*100, 1))+' p.p.'
cols = st.columns(3)
with cols[0]:
    st.metric(label="Dólar Blue Hoy",
          value='$' + str(round(df['venta_informal'].iloc[-1])),
          delta=aumento_porcentaje(df['informal_ajustado'].iloc[-1], df['informal_ajustado'].iloc[-2]),
          delta_color='inverse')
with cols[1]:
    st.metric(label=f"Inflación estimada de {calendar.month_name[pd.to_datetime('today').date().month]}",
          value=aumento_porcentaje(df['inflacion_arg'].iloc[-1]**30.5, 1),
          delta=aumento_porcentaje(df['inflacion_arg'].iloc[-1]**30.5, df['inflacion_arg'].resample('m').first().iloc[-2]**30.5, puntos_porcentuales=True),
          delta_color='inverse',
          help='Relevamiento de Expectativas de Inflación del BCRA')
with cols[2]:
    st.metric(label=f"Equivalente a fin de {calendar.month_name[pd.to_datetime('today').date().month]}",
          value='$' + str(round(df['venta_informal'].iloc[-1]*df['inflacion_arg'].iloc[-1]**(30.5-df.index[-1].day))),
              help='Este sería el valor del dólar blue a fin de mes si mantuviera su valor real, asumiendo que se cumple la expectativa de inflación, y que la inflación es homogénea a lo largo del mes.')


    ajustador = (df_grafico.inflacion_arg[::-1].cumprod() / df_grafico.inflacion_us[::-1].cumprod()).shift(1, fill_value=1)
    df_grafico['informal_ajustado'] = df_grafico['informal_ajustado'] / ajustador[fecha]

    fig = px.line(df_grafico.reset_index(), x='fecha', y='informal_ajustado', hover_data=['venta_informal'],
                  log_y=True, title=f'Precio Dólar Blue a pesos de {fecha.date()}')

    # Add vertical line for each year
    for year in df_grafico.index.year.unique():
        fig.add_vline(x=pd.Timestamp(year, 1, 1), name=f"Vertical Line {year}", line_width=0.05)

    # Extend range_x limit a bit further than the current one
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.expander(label='Data', expanded=False):
        st.dataframe(df_grafico.iloc[::-1])

# idx_fecha = len(df) - 1
fecha = st.slider('Seleccioná una fecha', df.index.min().date(), df.index.max().date(), df.index.max().date(), format="DD/MM/YY")
update_plot(fecha=fecha)

# Footer with GitHub link
st.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/LeoArtaza/precio-dolar-real)
    """,
    unsafe_allow_html=True,
)