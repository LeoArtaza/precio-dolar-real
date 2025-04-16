from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Precio Dólar Real", page_icon="📈")

st.title("Precio Dólar Real")

ajustador = lambda x: (x.inflacion_arg[::-1].cumprod() / x.inflacion_us[::-1].cumprod()).shift(1, fill_value=1)
@st.cache_data(ttl=pd.Timedelta(hours=1))
def cargar_datos():
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(index_col=0, parse_dates=True)


    # Crear valores ajustados
    df['informal_ajustado'] = df['venta_informal']*ajustador(df)
    df['oficial_ajustado'] = df['venta_oficial']*ajustador(df)
    
    return df

df = cargar_datos()

@st.cache_data(ttl=pd.Timedelta(minutes=15))
def cargar_dolar_hoy():
    try:
        r = requests.get('https://dolarapi.com/v1/dolares/oficial')
        dolar_hoy = eval(r.text)
        df.loc[df.index[-1], ['venta_informal', 'informal_ajustado']] = dolar_hoy['venta']
    except Exception as e:
        st.warning('No se pudo acceder al valor actual.', e)
    return df

df = cargar_dolar_hoy()

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
          delta=aumento_porcentaje(df['inflacion_arg'].iloc[-1]**30.5, df['inflacion_arg'].resample('ME').first().iloc[-2]**30.5, puntos_porcentuales=True),
          delta_color='inverse',
          help='Relevamiento de Expectativas de Inflación del BCRA')
with cols[2]:
    st.metric(label=f"Equivalente a fin de {calendar.month_name[pd.to_datetime('today').date().month]}",
          value='$' + str(round(df['venta_informal'].iloc[-1]*df['inflacion_arg'].iloc[-1]**(30.5-df.index[-1].day))),
              help='Este sería el valor del dólar blue a fin de mes si mantuviera su valor real, asumiendo que se cumple la expectativa de inflación, y que la inflación es homogénea a lo largo del mes.')

st.divider()

preset_fecha_dict = {'3m': pd.Timedelta(days=90),
                     '6m': pd.Timedelta(days=180),
                     '1a': pd.Timedelta(days=365),
                     '2a': pd.Timedelta(days=365.25*2),
                     '5a': pd.Timedelta(days=365.25*5),
                     '10a': pd.Timedelta(days=365.25*10),
                     '20a': pd.Timedelta(days=365.25*20),
                     'Máx.': pd.Timedelta(days=len(df)-1)}

fig_container = st.container()

with fig_container:
    preset_fecha = st.radio('Rangos de fechas predeterminados', list(preset_fecha_dict.keys())[::-1], index=2, key='preset_fecha',
                        horizontal=True, label_visibility='collapsed')

with st.expander(label='Opciones Avanzadas', expanded=False):
    rango_fecha = st.slider('Rango de fechas', df.index.min().date(), df.index.max().date(),
                            value=(df.index.max().date() - preset_fecha_dict[preset_fecha], df.index.max().date()),
                            format="DD/MM/YY", key='slider_fechas')
    cols = st.columns(spec=[0.2, 1])
    with cols[0]:
        link_precio_rango = st.toggle(label='🔗', help='La fecha de referencia de precios será el inicio del gráfico.', key='link_precio_rango')
    with cols[1]:
        base_100 = st.toggle(label='Base 100')
        
    if link_precio_rango:
        st.session_state['fecha_precio_referencia'] = st.session_state['slider_fechas'][0]
    else:
        if 'fecha_precio_referencia' not in st.session_state:
            st.session_state['fecha_precio_referencia'] = df.index.max().date()
        else:
            st.session_state['fecha_precio_referencia'] = st.session_state['fecha_precio_referencia']

    fecha_precio_referencia = st.slider('Fecha de referencia de precios', df.index.min().date() , df.index.max().date(), format="DD/MM/YY", key='fecha_precio_referencia')

fecha_precio_referencia = pd.to_datetime(fecha_precio_referencia)


df['informal_ajustado_a_fecha'] = (df['informal_ajustado'] / ajustador(df)[fecha_precio_referencia]).round(2)
df['oficial_ajustado_a_fecha'] = (df['oficial_ajustado'] / ajustador(df)[fecha_precio_referencia]).round(2)

nombre_variable = 'Ajustado informal'
if base_100:
    df['informal_ajustado_a_fecha'] /= df.loc[fecha_precio_referencia, 'informal_ajustado_a_fecha'] * 0.01
    df['oficial_ajustado_a_fecha' ] /= df.loc[fecha_precio_referencia,  'oficial_ajustado_a_fecha'] * 0.01
    nombre_variable = 'Índice de precio'

fig = px.line(df.reset_index().rename(columns={'fecha': 'Fecha', 'venta_informal': 'Venta informal', 'informal_ajustado_a_fecha': nombre_variable, 'oficial_ajustado_a_fecha': 'Ajustado oficial'}),
              x='Fecha', y=nombre_variable, hover_data=['Fecha', 'Venta informal', nombre_variable], log_y=True,
              title='Precio del dólar' + (f' a pesos de {fecha_precio_referencia.strftime("%d de %B de %Y")}' if not base_100 else f'. Base 100 = {fecha_precio_referencia.date()}'))

# Add Dólar Oficial trace
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['oficial_ajustado_a_fecha'],
    mode='lines',
    name='Ajustado oficial', # Name for hover
    line=dict(color='orange'), # Optional: Set a distinct color
    hovertemplate='<b>Fecha</b>: %{x|%d/%m/%Y}<br><b>Ajustado oficial</b>: %{y:.2f}<extra></extra>', # Custom hover text
    showlegend=False # Hide this trace from the legend
))

# --- Add Vertical Lines for Presidential Terms using Shapes ---
presidencies = [
    {"start": "1989-12-10", "color": "rgb(173, 216, 230)", "name": "Menem"},
    {"start": "1999-12-10", "color": "rgb(255, 0, 0)", "name": "De La Rúa"},
    {"start": "2001-12-21", "color": "rgb(173, 216, 230)", "name": "Duhalde"},
    {"start": "2003-05-25", "color": "rgb(173, 216, 230)", "name": "Kirchner"},
    {"start": "2007-12-10", "color": "rgb(173, 216, 230)", "name": "CFK"},
    {"start": "2011-12-10", "color": "rgb(173, 216, 230)", "name": "CFK2"},
    {"start": "2015-12-10", "color": "rgb(255, 215, 0)", "name": "Macri"},
    {"start": "2019-12-10", "color": "rgb(173, 216, 230)", "name": "Alberto"},
    {"start": "2023-12-10", "color": "rgb(128, 0, 128)", "name": "Milei"}
]

min_date = df.index.min()
max_date = df.index.max()

for pres in presidencies:
    start_date = pd.Timestamp(pres["start"])

    # Only add line if start date is within the data range
    if start_date >= min_date and start_date <= max_date:
        # Add the vertical line shape
        fig.add_shape(
            type="line",
            xref="x", yref="paper", # x=date axis, y=full plot height
            x0=start_date, y0=0,    # Start at the date, bottom of plot
            x1=start_date, y1=1,    # End at the date, top of plot
            line=dict(
                color=pres["color"],
                width=1.5,
                dash="dash",
            ),
            layer="below" # Draw below data lines
        )
# --- End Vertical Lines ---


if fecha_precio_referencia != df.index[-1]:
    # Linea en fecha de referencia
    fig.add_vline(x=fecha_precio_referencia, line_dash="dash", name="Fecha precio de referencia", line_width=1, line_color='gray')
    # Annotation en fecha de referencia
    fig.add_annotation(
        x=fecha_precio_referencia,
        y=np.log10(df['informal_ajustado_a_fecha'].loc[fecha_precio_referencia]),
        xref="x",
        yref="y",
        text=str(np.round(df['informal_ajustado_a_fecha'].loc[fecha_precio_referencia], 2)),
        font=dict(
            size=12,
            color="#ffffff",
            ),
        xanchor="left",
        yanchor="bottom",
        borderpad=1,
        bgcolor="rgb(25, 94, 221)",
        opacity=0.8,
        showarrow=True,
        arrowcolor="rgba(0, 0, 0, 0)",
        ax=5,
        ay=-3,
        )

# Línea horizontal en precio actual
fig.add_hline(y=df['informal_ajustado_a_fecha'].iloc[-1], name="Precio actual", line_dash="dash",
              line_width=0.5, line_color='gray', annotation_text='Precio actual', annotation_position='top left',
              annotation_font_size=150,
              annotation_font_color="blue")

# Add vertical line for each year
for year in df.index.year.unique():
    fig.add_vline(x=pd.Timestamp(year, 1, 1), name=f"Año {year}", line_width=0.05)

# Extend range_x limit a bit further than the current one
df_filtrado = df.loc[rango_fecha[0]:rango_fecha[1], 'informal_ajustado_a_fecha']
x_padding = pd.Timedelta(days=len(df_filtrado)//13)
y_padding = 1.1
fig.update_xaxes(range=[rango_fecha[0], rango_fecha[1] + x_padding], showspikes=True, spikethickness=0.5)
fig.update_yaxes(range=[np.log10(df_filtrado.min()/y_padding),
                        np.log10(df_filtrado.max()*y_padding)], type="log", showspikes=True, spikethickness=0.5)

fig.add_annotation(text="dolar-real.streamlit.app",
                  xref="paper", yref="paper",
                  x=1, y=0, showarrow=False, align="right")

# Annotation en fecha de hoy
fig.add_annotation(
    x=df.index[-1],
    y=np.log10(df['informal_ajustado_a_fecha'].iloc[-1]),
    xref="x",
    yref="y",
    text=str(np.round(df['informal_ajustado_a_fecha'].iloc[-1], 2)),
    font=dict(
        size=12,
        color="#ffffff",
        ),
    xanchor="left",
    yanchor="bottom",
    borderpad=1,
    bgcolor="rgb(25, 94, 221)",
    opacity=0.8,
    showarrow=True,
    arrowcolor="rgba(0, 0, 0, 0)",
    ax=5,
    ay=-3,
    )

fig.update_layout(dragmode=False, xaxis_title='Fecha', yaxis_title=nombre_variable,
                  hoverlabel=dict(bgcolor="rgba(25, 94, 221, 0.8)", font_color="white"))

with fig_container:
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with st.expander(label='Metodología', expanded=False):
    st.markdown("""## Cálculo
Para poder calcular el precio real del dólar de manera diaria, se debe estimar la inflación diaria a partir de la inflación mensual.
Para esto, se asume que la inflación es homogénea a la largo del mes, y se calcula de la siguiente manera:
""")
    st.latex(r'\text{Inflación diaria} = (1 + \text{Inflación mensual})^{\frac{1}{\text{ctdad. días en mes}}}')
    st.markdown("""
Luego, se calcula el valor del dólar ajustado por inflación del peso argentino, e inflación del dólar estadounidense:
""")
    st.latex(r'\text{Precio Dólar Real}_t = \text{Precio Dólar}_t \times \frac{\prod_{i=1}^{t} (1 + \text{Inflación diaria del peso}_i)}{\prod_{i=1}^{t} (1 + \text{Inflación diaria del dólar}_i)}')
    st.markdown("""Donde 't' es la cantidad de días en el pasado que se quiere calcular.
## Fuentes:
- Dólar
    - Precio oficial
        - Ene 1992-Abr 2002: Datos.gob.ar, serie: 175.1_DR_ESTANSE_0_0_20
        - Abr 2002-Presente: Ámbito Financiero
    - Precio blue
        - Ene 1992-Abr 2002: Dólar blue = Dólar oficial
        - Abr 2002-Presente: Ámbito Financiero
        - Día de hoy: dolarapi.com
- Inflación
    - Inflación Argentina
        - 1992-2017: inflacionverdadera.com/argentina
        - 2017-Presente: Datos.gob.ar, serie: 148.3_INIVELNAL_DICI_M_26
    - Inflación EEUU
        - fred.stlouisfed.org, serie: CPIAUCNS
""")

# GitHub link
st.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/LeoArtaza/precio-dolar-real)
    """,
    unsafe_allow_html=True,
)