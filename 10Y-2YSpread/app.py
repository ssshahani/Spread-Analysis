'''

Implemented by Sahil Shahani and Assistant: AI (Claude)
Date: 31 August 2025
'''

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple
import os
import io
import plotly.graph_objects as go
import pandas_datareader as web
from fredapi import Fred
from configparser import ConfigParser

st.set_page_config(page_title="10y-2y Yield Spread", page_icon=":chart_with_upwards_trend:", layout="wide")
st.sidebar.title("Settings")

default_years=15
years_back = st.sidebar.slider("Window(years)", min_value=2, max_value=60, value=default_years, step=1)
show_recessions = st.sidebar.checkbox("Shade US Recessions", value=True)
show_inversions = st.sidebar.checkbox("Highlight invesrion periods", value=True)
smooth_days = st.sidebar.slider("Optional smoothing days (rolling mean", 0,60,0, help="Set >0 to smooth the spread series. ")
st.sidebar.markdown("---")
st.sidebar.caption("Data Source:: FRED (DGS10, DGS2, USREC)")

config_object = ConfigParser()
inifile_read = config_object.read('config.ini')
if not inifile_read:
    print("Error: The config.ini file was not found or could not be read.")
    print(f"current working dir {os.getcwd()}")
else:
    print("Successfully read config file(s):", inifile_read)

# This is the key troubleshooting step
print("Sections found:", config_object.sections())

try:
    api_key_section = config_object['API_KEY']
    fred_api_key = api_key_section["fred_api_key"]
    print("Success! FRED API Key:", fred_api_key)
except KeyError as e:
    print(f"KeyError: {e} - The section was not found.")

@st.cache_data
def fetch_fred_series(series_id) -> pd.Series:
    """

    :rtype: object
    """
    fred = Fred(api_key=fred_api_key)
    s = fred.get_series(series_id)
    #s = web.get_data_fred(str(series_id)) # returns a pandas series indexed by date
    #s = web.get_data_fred(np.array([series_id]))
    s.name = series_id
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = pd.to_numeric(s, errors="coerce")
    return s


@st.cache_data
def load_data() -> pd.DataFrame:
    y10 = fetch_fred_series('DGS10')
    y2 = fetch_fred_series('DGS2')
    rec = fetch_fred_series('USREC')
    df = pd.concat([y10, y2, rec], axis=1).rename(columns={"DGS2": "2Y", "DGS10": "10Y", "USREC": "Recession"})
    df['Spread'] = df["10Y"] - df["2Y"]
    df = df.dropna(subset=["10Y", "2Y", "Recession"])
    return df


def filter_by_years(df, years) -> pd.DataFrame:
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years = years)
    return df[start:end]


def find_inversion_periods(spread: pd.Series, min_days=1) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """ Return list of (start, end) when spread < 0. min_days merges micro gaps"""
    below = spread<0
    if below.sum() == 0:
        return []
    # identify contiguous regions
    runs = []
    start = None
    prev = None
    for date, is_below in below.items():
        if is_below and start is None: # checking if is_below is True and start is None
            start = date
        if not is_below and start is not None: #checking if is_below is False and start is not None
            end = prev
            runs.append([start, end])
            start = None
        prev = date

    # if current period is inversion (maybe that's why)
    if start is not None:
        runs.append([start, prev])

    merged = []

    # Optional: merge brief interruptions (gap of 1 day)
    for s,e in runs:
        if not merged:
            merged.append([s,e])
        else:
            last_s, last_e = merged[-1]
            if (s - last_e) <= pd.Timedelta(days=min_days):
                merged[-1][1] = e
            else:
                merged.append([s,e])

    return [(s,e) for s,e in merged]


def summarize_stats(df) -> pd.DataFrame:
    latest_date = df.index.max()
    latest_row = df.loc[latest_date]
    out = {
        "Latest Date": [latest_date.date()],
        "10Y (%)": [latest_row["10Y"]],
        "2Y (%)": [latest_row["2Y"]],
        "Spread (pp)": [latest_row["Spread"]],
        "Min Spread (pp)": [df["Spread"].min()],
        "Max Spread (pp)": [df["Spread"].max()],
        "Avg Spread (pp)": [df["Spread"].mean()],
        "Percentile (pp)": [df["Spread"].rank(pct=True).iloc[-1] * 100]
    }

    return pd.DataFrame(out)


# Main
st.title(" 10Y -2Y Treasury Yield Spread ")
st.caption(" 10 Year minus 2Y Treasury Yields - from FRED")

with st.spinner("Loading data from FRED .."):
    try:
        df_all = load_data()
    except Exception as e:
        st.error(f"Failed to load data from FRED error: {e}")
        st.stop()

if df_all.empty:
    st.error("No data returned from the FRED. ")
    st.stop()

#Apply smoothing if requested

df_all_proc = df_all.copy()
if smooth_days and smooth_days>0:
    df_all_proc["Spread"] = df_all_proc["Spread"].rolling(smooth_days, min_periods=1).mean()

df = filter_by_years(df_all_proc, years_back)

col1, col2 = st.columns([3,2], vertical_alignment="top")

with col1:
    fig = go.Figure()
    # Spread Line
    fig.add_trace(go.Scatter(x=df.index, y=df["Spread"], mode="lines", name="10y-2y spread (pp)"))
    #Zero Line
    fig.add_hline(y=0, line_dash="dash")

    # Show Recessions
    if show_recessions and "Recession" in df.columns:
        rec = df_all.loc[df.index.min():df.index.max(), "Recession"].fillna(0)
        # Find spans where recession = 1
        in_rec = False
        rec_start = None
        for date, val in rec.items():
            if val == 1 and not in_rec:
                in_rec = True
                rec_start = date
            if (val == 0 or pd.isna(val)) and in_rec:
                fig.add_vrect(x0=rec_start, x1=date, fillcolor="red", opacity=0.3, layer ="below", line_width = 0)
                in_rec = False

        if in_rec:
            fig.add_vrect(x0=rec_start, x1=rec.index.max(),fillcolor="red", opacity=0.3, layer ="below",
                          line_width = 0)

    # Show Inversions:
    if show_inversions:
        periods = find_inversion_periods(df["Spread"])
        for s, e in periods:
            fig.add_vrect(x0=s, x1=e, fillcolor="salmon", opacity=0.2, layer="below", line_width = 0)


    fig.update_layout(
        title = f"10y-2y Spread {years_back} years",
        xaxis_title = "Date",
        yaxis_title = "Percentage Points",
        hovermode = "x unified",
        height = 520,
        margin = dict(l=40,r=20,t=60 , b =40)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Latest & Summary")
    st.dataframe(summarize_stats(df_all).style.format({
        "10Y (%)" : "{:.2f}", "2Y (%)" : "{:.2f}", "Spread (pp)": "{:.2f}",
        "Percentile (since 1976)": "{:.1f}%",
        "Min Spread (pp)": "{:.1f}%", "Max Spread (pp)": "{:.1f}%", "Avg Spread (pp)": "{:.1f}%"
    }), use_container_width=True, hide_index=True)

    st.markdown("--")
    st.subheader("Inversion Episodes (filtered window)")
    inv_periods = find_inversion_periods(df["Spread"])
    if inv_periods:
        inv_df = pd.DataFrame([{"Start": s.date(), "End": e.date(), "Days" : (e-s).days + 1} for s, e in inv_periods])
        st.dataframe(inv_df, use_container_width=True, hide_index=True)
    else:
        st.info("No inversion periods in current window")

#Raw yields chart
st.markdown("### 10y and 2y Yields")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["10Y"], mode="lines", name="10y"))
fig2.add_trace(go.Scatter(x=df.index, y=df["2Y"], mode="lines", name="2y"))
fig2.update_layout(hovermode = "x unified", xaxis_title="Date", yaxis_title="Percent", height = 380,
                   margin = dict(l=40,r=20,t=30 , b =40))
st.plotly_chart(fig2, use_container_width=True)