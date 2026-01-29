import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Rohstoffpreise – Band / Whisker", layout="wide")

# ===================== Helpers =====================
def read_any_csv(file_obj) -> pd.DataFrame:
    try:
        df_ = pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        file_obj.seek(0)
        df_ = pd.read_csv(file_obj, sep=";")
    df_.columns = [c.replace("\ufeff", "").strip() for c in df_.columns]
    return df_

def to_num_series(s: pd.Series) -> pd.Series:
    # robust parsing: handles numbers, "123,4", "123.4", "€", spaces
    if s.dtype == object:
        s = (
            s.astype(str)
             .str.replace("€", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(s, errors="coerce")

def normalize_colname(c: str) -> str:
    # keep it simple; match headers like "Gas Low €/MWh" etc.
    return (
        c.strip()
         .replace("€", "")
         .replace("/MWh", "")
         .replace("/t", "")
         .replace("/a", "")
         .replace("  ", " ")
    )

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"Keine passende Spalte gefunden. Gesucht: {candidates}. Gefunden: {list(df.columns)}")

def calc_q_from_low_mid_high(low: pd.Series, mid: pd.Series, high: pd.Series):
    """
    You want a 5-value whisker per year built from:
    [Low, (Low+Mid)/2, Mid, (Mid+High)/2, High]
    Then the whisker stats are:
      min=Low, max=High, q1=(Low+Mid)/2, median=Mid, q3=(Mid+High)/2
    """
    q1 = (low + mid) / 2
    med = mid
    q3 = (mid + high) / 2
    return low, q1, med, q3, high

def add_band(fig, x, low, high, name, fill_rgba, line_rgba, dashed=True):
    xs = pd.concat([x, x[::-1]])
    ys = pd.concat([high, low[::-1]])
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill="toself",
        mode="lines",
        name=name,
        line=dict(color=line_rgba, width=1, dash="dash" if dashed else "solid"),
        fillcolor=fill_rgba,
        hovertemplate="skip"
    ))

def add_whisker_panel(fig, row, x, vmin, q1, med, q3, vmax, title, color):
    # One panel, many years: draw per-year whisker + IQR box + median tick
    for xi, lo, a, b, c, hi in zip(x, vmin, q1, med, q3, vmax):
        # whisker
        fig.add_trace(go.Scatter(
            x=[xi, xi], y=[lo, hi],
            mode="lines",
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo="skip"
        ), row=row, col=1)

        # caps
        fig.add_trace(go.Scatter(
            x=[xi - 0.18, xi + 0.18], y=[lo, lo],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip"
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=[xi - 0.18, xi + 0.18], y=[hi, hi],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip"
        ), row=row, col=1)

        # IQR box (q1->q3)
        fig.add_trace(go.Scatter(
            x=[xi-0.25, xi+0.25, xi+0.25, xi-0.25, xi-0.25],
            y=[a, a, c, c, a],
            fill="toself",
            mode="lines",
            line=dict(color=color, width=1),
            fillcolor=color.replace("1.0", "0.18") if "rgba" in color else color,
            showlegend=False,
            hoverinfo="skip"
        ), row=row, col=1)

        # median tick
        fig.add_trace(go.Scatter(
            x=[xi - 0.25, xi + 0.25], y=[b, b],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=f"Jahr={xi}<br>{title} Median=%{{y:.2f}}<extra></extra>"
        ), row=row, col=1)

    fig.update_yaxes(title_text=title, row=row, col=1)

# ===================== UI =====================
st.title("Rohstoffpreise: Band (Low–High) ↔ Whisker (aus Low/Mid/High) + GuV Szenarien")

uploaded = st.file_uploader("CSV hochladen", type=["csv"])
if uploaded is None:
    st.info("Bitte CSV hochladen.")
    st.stop()

df = read_any_csv(uploaded)

# Normalize and numeric-parse
df.columns = [normalize_colname(c) for c in df.columns]
for c in df.columns:
    if c.lower() != "jahr":
        df[c] = to_num_series(df[c])

df["Jahr"] = to_num_series(df["Jahr"]).astype("Int64")
df = df.dropna(subset=["Jahr"]).sort_values("Jahr")
x = df["Jahr"].astype(int)

# ===================== Column mapping (works with your shown header) =====================
# Expected columns (after normalize_colname):
# Jahr
# Low MWh / Mid MWh / High MWh (twice) -> NOT unique in your current header.
# Therefore: you MUST have unique headers OR we match by position. We'll match by POSITION safely:
#
# Your order (as you posted):
# Jahr;
# 1) Gas Low €/MWh; 2) Gas Mid €/MWh; 3) Gas High €/MWh;
# 4) Strom Low €/MWh; 5) Strom Mid €/MWh; 6) Strom High €/MWh;
# 7) CO2 Low €/t; 8) CO2 Mid €/t; 9) CO2 High €/t;
# 10) GuV S1 Best €/a; 11) GuV S1 Worst €/a; 12) GuV S2 Best €/a; 13) GuV S2 Worst €/a
#
# We'll use index-based extraction to avoid ambiguous duplicate header names.
cols = list(df.columns)
if len(cols) < 14:
    st.error(f"CSV hat zu wenige Spalten ({len(cols)}). Erwartet ~14 inkl. Jahr. Spalten: {cols}")
    st.stop()

# index mapping by position
gas_low, gas_mid, gas_high = cols[1], cols[2], cols[3]
strom_low, strom_mid, strom_high = cols[4], cols[5], cols[6]
co2_low, co2_mid, co2_high = cols[7], cols[8], cols[9]
guv1_best, guv1_worst, guv2_best, guv2_worst = cols[10], cols[11], cols[12], cols[13]

# ===================== Colors (simple) =====================
C_GAS_FILL   = "rgba(120, 180, 255, 0.25)"  # light blue
C_GAS_LINE   = "rgba(120, 180, 255, 0.90)"
C_STROM_FILL = "rgba(255, 165, 0, 0.22)"    # orange
C_STROM_LINE = "rgba(255, 165, 0, 0.90)"
C_CO2_FILL   = "rgba(160, 160, 160, 0.18)"  # grey
C_CO2_LINE   = "rgba(160, 160, 160, 0.85)"
C_GUV_S1     = "rgba(220, 60, 60, 0.85)"   # rot
C_GUV_S2     = "rgba(40, 170, 90, 0.85)"   # grün

# ===================== Mode =====================
mode = st.sidebar.radio("Darstellung", ["Band (Low–High)", "Whisker (5 Werte aus Low/Mid/High)"])

show_guv = st.sidebar.checkbox("GuV anzeigen", True)

# ===================== Plot =====================
if mode == "Band (Low–High)":
    # Bands for commodities (Gas/Strom/CO2) using Low/High
    fig = go.Figure()

    add_band(fig, x, df[gas_low], df[gas_high], "Gas (Low–High, €/MWh)", C_GAS_FILL, C_GAS_LINE, dashed=True)
    add_band(fig, x, df[strom_low], df[strom_high], "Strom (Low–High, €/MWh)", C_STROM_FILL, C_STROM_LINE, dashed=True)
    add_band(fig, x, df[co2_low], df[co2_high], "CO₂ (Low–High, €/t)", C_CO2_FILL, C_CO2_LINE, dashed=True)

    # GuV scenarios separated: 2 ranges on y2 (right axis)
    if show_guv:
        # Scenario 1 range (Worst–Best)
        fig.add_trace(go.Bar(
            x=x,
            y=df[guv1_best] - df[guv1_worst],
            base=df[guv1_worst],
            name="GuV S1 (Worst–Best)",
            yaxis="y2",
            opacity=0.35,
            marker=dict(color="rgba(40, 170, 90, 0.85)hh"),
            hovertemplate="Jahr=%{x}<br>S1 Worst=%{base:.0f}<br>S1 Best=%{y+base:.0f}<extra></extra>",
        ))
        # Scenario 2 range (Worst–Best)
        fig.add_trace(go.Bar(
            x=x,
            y=df[guv2_best] - df[guv2_worst],
            base=df[guv2_worst],
            name="GuV S2 (Worst–Best)",
            yaxis="y2",
            opacity=0.25,
            marker=dict(color="rgba(220,60,60,0.35)"),
            hovertemplate="Jahr=%{x}<br>S2 Worst=%{base:.0f}<br>S2 Best=%{y+base:.0f}<extra></extra>",
        ))

    fig.update_layout(
        height=740,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Jahr", showgrid=False),

        yaxis=dict(
            title="Preise (Gas/Strom: €/MWh, CO₂: €/t)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)"
        ),
        yaxis2=dict(
            title="GuV (€ / a)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False
        ),
        barmode="overlay"
    )

    # GuV Null-Linie
    if show_guv:
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y2", y0=0, y1=0,
            line=dict(color="rgba(220,60,60,0.7)", width=1, dash="dot")
        )

else:
    # Whisker per year, 4 panels: Gas / Strom / CO2 / GuV (split into two panels? user wants separated)
    # You asked: "GuV 1 und GuV 2 jeweils mit best und worsed werten" => show as two panels.
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # Gas 5-value whisker from Low/Mid/High:
    gmin, gq1, gmed, gq3, gmax = calc_q_from_low_mid_high(df[gas_low], df[gas_mid], df[gas_high])
    add_whisker_panel(fig, 1, x, gmin, gq1, gmed, gq3, gmax, "Gas (€/MWh)", "rgba(120, 180, 255, 1.0)")

    # Strom
    smin, sq1, smed, sq3, smax = calc_q_from_low_mid_high(df[strom_low], df[strom_mid], df[strom_high])
    add_whisker_panel(fig, 2, x, smin, sq1, smed, sq3, smax, "Strom (€/MWh)", "rgba(255, 165, 0, 1.0)")

    # CO2 (€/t)
    cmin, cq1, cmed, cq3, cmax = calc_q_from_low_mid_high(df[co2_low], df[co2_mid], df[co2_high])
    add_whisker_panel(fig, 3, x, cmin, cq1, cmed, cq3, cmax, "CO₂ (€/t)", "rgba(160, 160, 160, 1.0)")

    # GuV S1 (Worst/Best only) -> build 5-values via the same scheme BUT WITHOUT mid in file:
    # You asked to use best/worst and still create 5 values for whisker:
    # We'll define:
    #   min = worst
    #   q1  = worst + 0.25*(best-worst)
    #   med = worst + 0.50*(best-worst)
    #   q3  = worst + 0.75*(best-worst)
    #   max = best
    def guv_5(worst: pd.Series, best: pd.Series):
        vmin = worst
        vmax = best
        q1 = worst + 0.25 * (best - worst)
        med = worst + 0.50 * (best - worst)
        q3 = worst + 0.75 * (best - worst)
        return vmin, q1, med, q3, vmax

    if show_guv:
        vmin, q1, med, q3, vmax = guv_5(df[guv1_worst], df[guv1_best])
        add_whisker_panel(fig, 4, x, vmin, q1, med, q3, vmax, "GuV S1 (€/a)", "rgba(220, 60, 60, 1.0)")

        vmin, q1, med, q3, vmax = guv_5(df[guv2_worst], df[guv2_best])
        add_whisker_panel(fig, 5, x, vmin, q1, med, q3, vmax, "GuV S2 (€/a)", "rgba(220, 60, 60, 0.75)")
    else:
        fig.update_yaxes(title_text="GuV ausgeblendet", row=4, col=1)
        fig.update_yaxes(title_text="GuV ausgeblendet", row=5, col=1)

    fig.update_layout(
        height=1100,
        showlegend=False,
        xaxis=dict(title="Jahr", showgrid=False),
        template="plotly_white"
    )
    fig.update_xaxes(dtick=1)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Datenvorschau"):
    st.dataframe(df, use_container_width=True)
