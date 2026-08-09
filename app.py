from streamlit_gsheets import GSheetsConnection
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import requests

from bandas_cambiarias import (
    BandasCambiarias,
    InflacionConRezago,
    PorcentajeFijo,
    TramoPolitica,
    inflacion_mensual_desde_factor_diario,
)

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
        r = requests.get('https://dolarapi.com/v1/dolares/oficial', timeout=30)
        r.raise_for_status()
        dolar_hoy = r.json()
        fecha_precio_actual = df['venta_informal'].last_valid_index()
        if fecha_precio_actual is None:
            raise ValueError('No hay una fecha actual de precios disponible.')
        df.loc[fecha_precio_actual, ['venta_informal', 'informal_ajustado']] = dolar_hoy['venta']
    except Exception:
        st.warning('No se pudo acceder al valor actual.')
    return df

df = cargar_dolar_hoy()
df_precios = df.loc[df['venta_informal'].notna()].copy()
fecha_precio_actual = df_precios.index[-1]
df_inflacion_hasta_hoy = df.loc[:fecha_precio_actual]

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
          value='$' + str(round(df_precios['venta_informal'].iloc[-1])),
          delta=aumento_porcentaje(df_precios['informal_ajustado'].iloc[-1], df_precios['informal_ajustado'].iloc[-2]),
          delta_color='inverse')
with cols[1]:
    st.metric(label=f"Inflación estimada de {calendar.month_name[pd.to_datetime('today').date().month]}",
          value=aumento_porcentaje(df_inflacion_hasta_hoy['inflacion_arg'].iloc[-1]**30.5, 1),
          delta=aumento_porcentaje(df_inflacion_hasta_hoy['inflacion_arg'].iloc[-1]**30.5, df_inflacion_hasta_hoy['inflacion_arg'].resample('ME').first().iloc[-2]**30.5, puntos_porcentuales=True),
          delta_color='inverse',
          help='Relevamiento de Expectativas de Inflación del BCRA')
with cols[2]:
    st.metric(label=f"Equivalente a fin de {calendar.month_name[pd.to_datetime('today').date().month]}",
          value='$' + str(round(df_precios['venta_informal'].iloc[-1]*df_inflacion_hasta_hoy['inflacion_arg'].iloc[-1]**(30.5-fecha_precio_actual.day))),
              help='Este sería el valor del dólar blue a fin de mes si mantuviera su valor real, asumiendo que se cumple la expectativa de inflación, y que la inflación es homogénea a lo largo del mes.')

st.divider()

preset_fecha_dict = {
    '3m': pd.Timedelta(days=90),
    '6m': pd.Timedelta(days=180),
    '1a': pd.Timedelta(days=365),
    '2a': pd.Timedelta(days=365.25 * 2),
    '5a': pd.Timedelta(days=365.25 * 5),
    '10a': pd.Timedelta(days=365.25 * 10),
    '20a': pd.Timedelta(days=365.25 * 20),
    'Máx.': pd.Timedelta(days=len(df_precios) - 1),
}

def render_chart():
    chart_df = df.copy()

    fig_container = st.container()
    with fig_container:
        st.radio(
            'Rangos de fechas predeterminados',
            list(preset_fecha_dict.keys())[::-1],
            index=2,
            key='preset_fecha',
            horizontal=True,
            label_visibility='collapsed',
        )

    with st.expander(label='Opciones Avanzadas', expanded=False):
        rango_fecha = st.slider('Rango de fechas', df_precios.index.min().date(), df_precios.index.max().date(),
                                value=((pd.Timestamp(df_precios.index.max().date()) - pd.Timedelta(days=365.25 * 5)).date(), df_precios.index.max().date()),
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
                st.session_state['fecha_precio_referencia'] = df_precios.index.max().date()
            else:
                st.session_state['fecha_precio_referencia'] = st.session_state['fecha_precio_referencia']

        fecha_precio_referencia = st.slider('Fecha de referencia de precios', df_precios.index.min().date() , df_precios.index.max().date(), format="DD/MM/YY", key='fecha_precio_referencia')

    fecha_precio_referencia = pd.to_datetime(fecha_precio_referencia)
    rango_fecha = tuple(pd.Timestamp(fecha) for fecha in rango_fecha)

    # --- Band inflation adjustment ---
    # The bands remain active until the last date with an inflation expectation.
    last_known_inf_date = df['inflacion_arg'].last_valid_index()
    if last_known_inf_date is None:
        raise ValueError('No hay expectativas de inflación disponibles para situar las bandas.')

    last_known_inf_date = pd.Timestamp(last_known_inf_date)
    full_ajustador = ajustador(df)


    adjust_factor_ref = full_ajustador[fecha_precio_referencia]
    chart_df['informal_ajustado_a_fecha'] = (chart_df['informal_ajustado'] / adjust_factor_ref).round(2)
    chart_df['oficial_ajustado_a_fecha'] = (chart_df['oficial_ajustado'] / adjust_factor_ref).round(2)

    nombre_variable = 'Ajustado informal'
    if base_100:
        chart_df['informal_ajustado_a_fecha'] /= chart_df.loc[fecha_precio_referencia, 'informal_ajustado_a_fecha'] * 0.01
        chart_df['oficial_ajustado_a_fecha'] /= chart_df.loc[fecha_precio_referencia, 'oficial_ajustado_a_fecha'] * 0.01
        nombre_variable = 'Índice de precio'

    df_precios_chart = chart_df.loc[chart_df['venta_informal'].notna()].copy()
    x_padding = pd.Timedelta(days=365)
    df_filtrado = df_precios_chart.loc[rango_fecha[0]:rango_fecha[1]]
    df_chart_visible = df_precios_chart

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_chart_visible.index,
        y=df_chart_visible['informal_ajustado_a_fecha'],
        mode='lines',
        name=nombre_variable,
        customdata=df_chart_visible[['venta_informal']].to_numpy(),
        hovertemplate=(
            '<b>Fecha</b>: %{x|%d/%m/%Y}<br>'
            '<b>Venta informal</b>: $%{customdata[0]:.2f}<br>'
            f'<b>{nombre_variable}</b>: %{{y:.2f}}<extra></extra>'
        ),
    ))

    # Add Dólar Oficial trace
    fig.add_trace(go.Scatter(
        x=df_chart_visible.index,
        y=df_chart_visible['oficial_ajustado_a_fecha'],
        mode='lines',
        name='Ajustado oficial', # Name for hover
        line=dict(color='orange'), # Optional: Set a distinct color
        hovertemplate='<b>Fecha</b>: %{x|%d/%m/%Y}<br><b>Ajustado oficial</b>: %{y:.2f}<extra></extra>', # Custom hover text
        showlegend=False # Hide this trace from the legend
    ))

    # --- Policy-driven currency bands ---
    band_start_date = pd.Timestamp('2025-04-15')
    band_end_date = None  # None keeps the band open through the expectations horizon.
    band_period_end_date = last_known_inf_date if band_end_date is None else pd.Timestamp(band_end_date)
    initial_lower = 1000
    initial_upper = 1400

    inflacion_mensual = inflacion_mensual_desde_factor_diario(df['inflacion_arg'])
    bandas = BandasCambiarias(
        fecha_inicio=band_start_date,
        piso_inicial=initial_lower,
        techo_inicial=initial_upper,
        tramos=(
            TramoPolitica(
                desde=band_start_date,
                hasta=pd.Timestamp('2026-01-01'),
                politica=PorcentajeFijo(piso_mensual=-0.01, techo_mensual=0.01),
            ),
            TramoPolitica(
                desde=pd.Timestamp('2026-01-01'),
                hasta=None,
                politica=InflacionConRezago(rezago_meses=2),
            ),
        ),
    )
    trayectoria_bandas = bandas.trayectoria(
        inflacion_mensual,
        fecha_fin=band_period_end_date,
    )

    adjust_factor_ref = full_ajustador.reindex([fecha_precio_referencia]).ffill().bfill().iloc[0]
    adjust_factors_start = full_ajustador.reindex(
        pd.DatetimeIndex(trayectoria_bandas['desde'])
    ).ffill().bfill()
    if pd.notna(adjust_factor_ref) and adjust_factor_ref != 0:
        trayectoria_bandas['piso_ajustado'] = (
            trayectoria_bandas['piso_nominal']
            * (adjust_factors_start.to_numpy() / adjust_factor_ref)
        )
        trayectoria_bandas['techo_ajustado'] = (
            trayectoria_bandas['techo_nominal']
            * (adjust_factors_start.to_numpy() / adjust_factor_ref)
        )
    else:
        trayectoria_bandas['piso_ajustado'] = trayectoria_bandas['piso_nominal']
        trayectoria_bandas['techo_ajustado'] = trayectoria_bandas['techo_nominal']

    for band in trayectoria_bandas.itertuples(index=False):
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=band.desde, y0=band.piso_ajustado,
            x1=band.hasta, y1=band.techo_ajustado,
            fillcolor="rgba(0, 128, 0, 0.3)",
            layer="below",
            line_width=0,
        )

    # --- End Policy-driven currency bands ---

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

    min_date = df_precios_chart.index.min()
    max_date = df_precios_chart.index.max()

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


    if fecha_precio_referencia != fecha_precio_actual:
        # Linea en fecha de referencia
        fig.add_vline(x=fecha_precio_referencia, line_dash="dash", name="Fecha precio de referencia", line_width=1, line_color='gray')
        # Annotation en fecha de referencia
        fig.add_annotation(
            x=fecha_precio_referencia,
            y=np.log10(df_precios_chart['informal_ajustado_a_fecha'].loc[fecha_precio_referencia]),
            xref="x",
            yref="y",
            text=str(np.round(df_precios_chart['informal_ajustado_a_fecha'].loc[fecha_precio_referencia], 2)),
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
    fig.add_hline(y=df_precios_chart['informal_ajustado_a_fecha'].iloc[-1], name="Precio actual", line_dash="dash",
                  line_width=0.5, line_color='gray', annotation_text='Precio actual', annotation_position='top left',
                  annotation_font_size=150,
                  annotation_font_color="blue")

    # Add vertical line for each year
    for year in df_precios_chart.index.year.unique():
        fig.add_shape(
            type="line",
            xref="x", yref="paper",
            x0=pd.Timestamp(year, 1, 1), y0=0,
            x1=pd.Timestamp(year, 1, 1), y1=1,
            line=dict(width=0.05),
        )


    # Extend range_x limit a bit further than the current one
    y_padding = 1.1
    fig.update_xaxes(
        range=[rango_fecha[0], rango_fecha[1] + x_padding],
        showspikes=True,
        spikethickness=0.5,
    )
    # --- Calculate Y-axis range considering adjusted bands ---

    # 1. Calculate min/max of the adjusted bands over the full band period.
    full_band_end_date = band_period_end_date
    full_band_dates = pd.date_range(start=band_start_date, end=full_band_end_date, freq='D')
    abs_min_adj_lower_band = np.inf
    abs_max_adj_upper_band = -np.inf
    temp_full_band_df = pd.DataFrame()

    if not full_band_dates.empty:
        temp_full_band_df = pd.DataFrame(index=full_band_dates)
        if not trayectoria_bandas.empty:
            band_axis_values = trayectoria_bandas.set_index('desde')[[
                'piso_nominal',
                'techo_nominal',
                'piso_ajustado',
                'techo_ajustado',
            ]]
            temp_full_band_df = band_axis_values.reindex(temp_full_band_df.index).ffill()
            abs_min_adj_lower_band = temp_full_band_df['piso_ajustado'].min()
            abs_max_adj_upper_band = temp_full_band_df['techo_ajustado'].max()

    # 2. Find min/max of adjusted bands *within the filtered date range* (for max calculation)
    band_dates_in_range = pd.date_range(start=max(band_start_date, pd.Timestamp(rango_fecha[0])),
                                        end=min(full_band_end_date, pd.Timestamp(rango_fecha[1])),
                                        freq='D')
    max_adj_band_in_range = -np.inf
    if not band_dates_in_range.empty and 'techo_ajustado' in temp_full_band_df.columns:
         max_adj_band_in_range = temp_full_band_df.loc[band_dates_in_range, 'techo_ajustado'].max()
    elif not band_dates_in_range.empty and 'techo_nominal' in temp_full_band_df.columns: # Fallback if adjustment failed
         max_adj_band_in_range = temp_full_band_df.loc[band_dates_in_range, 'techo_nominal'].max()

    # 3. Determine overall min/max y values for axis calculation
    # Min value considers the data in view AND the absolute minimum of the lower band
    min_y_data_in_view = df_filtrado['informal_ajustado_a_fecha'].min()
    if 'oficial_ajustado_a_fecha' in df_filtrado:
        min_y_data_in_view = min(min_y_data_in_view, df_filtrado['oficial_ajustado_a_fecha'].min())

    min_y_for_axis = min(min_y_data_in_view, abs_min_adj_lower_band if np.isfinite(abs_min_adj_lower_band) else min_y_data_in_view)

    # Max value considers the data in view AND the maximum of the band *within the view*
    max_y_data_in_view = df_filtrado['informal_ajustado_a_fecha'].max()
    if 'oficial_ajustado_a_fecha' in df_filtrado:
        max_y_data_in_view = max(max_y_data_in_view, df_filtrado['oficial_ajustado_a_fecha'].max())

    max_y_for_axis = max(max_y_data_in_view, max_adj_band_in_range if np.isfinite(max_adj_band_in_range) else max_y_data_in_view)

    # Calculate log range without padding first (handle potential log(0) or log(neg))
    # 4. Calculate log range without padding first
    log_min = np.log10(max(min_y_for_axis, 1e-9)) # Use small epsilon to avoid log(0)
    log_max = np.log10(max(max_y_for_axis, 1e-9))

    # Apply padding *after* taking the logarithm
    # Calculate padding in log scale (log(1.1) is approx 0.04)
    log_padding = np.log10(y_padding)
    padded_min_y_log = log_min - log_padding
    padded_max_y_log = log_max + log_padding

    fig.update_yaxes(range=[padded_min_y_log, padded_max_y_log], type="log", showspikes=True, spikethickness=0.5)
    # --- End Y-axis range update ---

    fig.add_annotation(text="dolar-real.streamlit.app",
                      xref="paper", yref="paper",
                      x=1, y=0, showarrow=False, align="right")

    # Annotation en fecha de hoy
    fig.add_annotation(
         x=fecha_precio_actual,
         y=np.log10(df_precios_chart['informal_ajustado_a_fecha'].iloc[-1]),
        xref="x",
        yref="y",
         text=str(np.round(df_precios_chart['informal_ajustado_a_fecha'].iloc[-1], 2)),
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

    fig.update_layout(
                      title=(
                          'Precio del dólar' +
                          (f' a pesos de {fecha_precio_referencia.strftime("%d de %B de %Y")}'
                           if not base_100 else f'. Base 100 = {fecha_precio_referencia.date()}')
                      ),
                      dragmode=False, xaxis_title='Fecha', yaxis_title=nombre_variable,
                      hoverlabel=dict(bgcolor="rgba(25, 94, 221, 0.8)", font_color="white"))

    with fig_container:
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


render_chart()

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