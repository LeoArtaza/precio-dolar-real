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

def update_plot(fecha_precio, rango_fecha, base_100=False):
    fecha_precio = pd.to_datetime(fecha_precio)

    df_grafico = df.copy()

    ajustador = (df_grafico.inflacion_arg[::-1].cumprod() / df_grafico.inflacion_us[::-1].cumprod()).shift(1, fill_value=1)

    df_grafico['informal_ajustado'] /= ajustador[fecha_precio]

    if base_100:
        df_grafico['informal_ajustado'] /= df_grafico.loc[fecha_precio, 'informal_ajustado'] * 0.01

    fig = px.line(df_grafico.reset_index(), x='fecha', y='informal_ajustado', hover_data=['venta_informal'],
                  log_y=True, title='Precio dólar blue' + (f' a pesos de {fecha_precio.strftime("%d de %B de %Y")}' if not base_100 else f'. Base 100 = {fecha_precio.date()}'))

    fig.add_vline(x=fecha_precio, line_dash="dash", name="Vertical Line", line_width=1)

    # Add vertical line for each year
    for year in df_grafico.index.year.unique():
        fig.add_vline(x=pd.Timestamp(year, 1, 1), name=f"Vertical Line {year}", line_width=0.05)

    # Extend range_x limit a bit further than the current one
    df_filtrado = df_grafico.loc[rango_fecha[0]:rango_fecha[1], 'informal_ajustado']
    x_padding = pd.Timedelta(days=len(df_filtrado)//20)
    y_padding = 1.1
    fig.update_xaxes(range=[rango_fecha[0], rango_fecha[1] + x_padding], showspikes=True, spikethickness=0.5)
    fig.update_yaxes(range=[np.log10(df_filtrado.min()/y_padding),
                            np.log10(df_filtrado.max()*y_padding)], type="log", showspikes=True, spikethickness=0.5)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.expander(label='Data', expanded=False):
        st.dataframe(df_grafico.iloc[::-1])

fecha_precio = st.slider('Fecha de referencia de precios', df.index.min().date(), df.index.max().date(), value=df.index.max().date(), format="DD/MM/YY", key='fecha_precio')

cols = st.columns(spec=[0.2, 1])
with cols[0]:
    link_precio_rango = st.toggle(label='🔗', help='La fecha de referencia de precios será el inicio del gráfico.')
with cols[1]:
    base_100 = st.toggle(label='Base 100')

rango_fecha = st.slider('Rango de fecha a visualizar', df.index.min().date(), df.index.max().date(), value=(df.index.max().date() - pd.Timedelta(days=365*5), df.index.max().date()), format="DD/MM/YY")
if link_precio_rango:
    fecha_precio = rango_fecha[0]

update_plot(fecha_precio=fecha_precio, rango_fecha=rango_fecha, base_100=base_100)

# Footer with GitHub link
st.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/LeoArtaza/precio-dolar-real)
    """,
    unsafe_allow_html=True,
)