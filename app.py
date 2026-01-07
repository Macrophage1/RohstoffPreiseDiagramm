import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Rohstoffpreise – Band / Whisker", layout="wide")
st.title("Rohstoffpreise: Banddarstellung ↔ Whisker-Plot")

# ===================== Upload =====================
uploaded = st.file_uploader(
    "CSV hochladen (Jahr;Gas_Low;Gas_High;Strom_Low;Strom_High;CO2_Low;CO2_High;GuV_Low;GuV_High)",
    type=["csv"]
)

def read_any_csv(file_obj) -> pd.DataFrame:
    try:
        df_ = pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        file_obj.seek(0)
        df_ = pd.read_csv(file_obj, sep=";")
    df_.columns = [c.replace("\ufeff", "").strip() for c in df_.columns]
    return df_

def to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace("€", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(s, errors="coerce")

if uploaded is None:
    st.info("Bitte CSV hochladen.")
    st.stop()

df = read_any_csv(uploaded)

required = [
    "Jahr",
    "Gas_Low", "Gas_High",
    "Strom_Low", "Strom_High",
    "CO2_Low", "CO2_High",
    "GuV_Low", "GuV_High"
]

for c in required:
    df[c] = to_num(df[c])

df = df.dropna(subset=required).sort_values("Jahr")
df["Jahr"] = df["Jahr"].astype(int)
x = df["Jahr"]

# ===================== Sidebar =====================
st.sidebar.header("Darstellung")
mode = st.sidebar.radio(
    "Plot-Typ",
    ["Preis-Bänder (Low–High)", "Whisker-Plot (5 Werte)"]
)

# ===================== Helfer =====================
def add_band(fig, x, low, high, name, color):
    xs = pd.concat([x, x[::-1]])
    ys = pd.concat([high, low[::-1]])
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill="toself",
        mode="lines",
        line=dict(width=1),
        fillcolor=color,
        name=name,
        hoverinfo="skip"
    ))

def add_whisker_panel(fig, row, x, low, high, title):
    q1 = low + 0.25 * (high - low)
    med = (low + high) / 2
    q3 = low + 0.75 * (high - low)

    for xi, lo, hi, a, b, c in zip(x, low, high, q1, med, q3):
        # Whisker
        fig.add_trace(go.Scatter(
            x=[xi, xi], y=[lo, hi],
            mode="lines", line=dict(width=1),
            showlegend=False
        ), row=row, col=1)

        # Box
        fig.add_trace(go.Scatter(
            x=[xi-0.25, xi+0.25, xi+0.25, xi-0.25, xi-0.25],
            y=[a, a, c, c, a],
            fill="toself",
            mode="lines",
            line=dict(width=1),
            showlegend=False
        ), row=row, col=1)

        # Median
        fig.add_trace(go.Scatter(
            x=[xi-0.25, xi+0.25],
            y=[b, b],
            mode="lines",
            line=dict(width=2),
            showlegend=False
        ), row=row, col=1)

    fig.update_yaxes(title_text=title, row=row, col=1)

# ===================== Plot =====================
if mode == "Preis-Bänder (Low–High)":
    fig = go.Figure()

    add_band(fig, x, df["Gas_Low"], df["Gas_High"], "Gas (€/MWh)", "rgba(0,120,255,0.4)")
    add_band(fig, x, df["Strom_Low"], df["Strom_High"], "Strom (€/MWh)", "rgba(255,140,0,0.4)")
    add_band(fig, x, df["CO2_Low"], df["CO2_High"], "CO₂ (€/t)", "rgba(160,160,160,0.4)")

    fig.add_trace(go.Bar(
        x=x,
        y=df["GuV_High"] - df["GuV_Low"],
        base=df["GuV_Low"],
        name="GuV (Worst–Best)",
        yaxis="y2",
        opacity=0.7
    ))

    fig.update_layout(
        height=720,
        xaxis_title="Jahr",
        yaxis=dict(title="Preis"),
        yaxis2=dict(
            title="GuV (€)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(orientation="h", y=1.05)
    )

else:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    add_whisker_panel(fig, 1, x, df["Gas_Low"], df["Gas_High"], "Gas (€/MWh)")
    add_whisker_panel(fig, 2, x, df["Strom_Low"], df["Strom_High"], "Strom (€/MWh)")
    add_whisker_panel(fig, 3, x, df["CO2_Low"], df["CO2_High"], "CO₂ (€/t)")
    add_whisker_panel(fig, 4, x, df["GuV_Low"], df["GuV_High"], "GuV (€)")

    fig.update_layout(
        height=950,
        xaxis_title="Jahr",
        showlegend=False
    )

st.plotly_chart(fig, use_container_width=True)

st.subheader("Datenvorschau")
st.dataframe(df, use_container_width=True)
