import streamlit as st
import plotly.express as px
import pandas as pd
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
        df.loc[dolar_blue_hoy['fechaActualizacion'].split('T')[0], ['venta_informal', 'informal_ajustado']] = dolar_blue_hoy['venta']
        return df
    except Exception as e:
        dolar_blue_hoy
        st.error('No se pudo acceder al valor del día de la fecha.', e)

df = get_data()

def update_plot(fecha):
    fecha = pd.to_datetime(fecha)

    df_grafico = df.copy()

    ajustador = (df_grafico.inflacion_arg[::-1].cumprod() / df_grafico.inflacion_us[::-1].cumprod()).shift(1, fill_value=1)
    df_grafico['informal_ajustado'] = df_grafico['informal_ajustado'] / ajustador[fecha]

    fig = px.line(df_grafico.reset_index(), x='fecha', y='informal_ajustado', hover_data=['venta_informal'],
                  log_y=True, title=f'Precio Dólar Blue a pesos de {fecha.date()}')

    fig.add_vline(x=fecha, line_dash="dash", name="Vertical Line", line_width=1)

    # Extend range_x limit a bit further than the current one
    fig.update_xaxes(range=[df_grafico.index.min(), df_grafico.index.max() + pd.Timedelta(days=365)])

    st.plotly_chart(fig)

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