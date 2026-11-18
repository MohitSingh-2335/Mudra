# src/ui/charts.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INITIAL_CAPITAL_USD = 10000.0
__all__ = ["create_tradingview_candlestick", "create_alpha_equity_curve", "create_reconciliation_scatter"]


def _base_layout(title: str, height: int = 380) -> dict:
    return dict(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=14, color="#EAECEF"), x=0.01, y=0.99),
        paper_bgcolor="#0E1117", plot_bgcolor="#131722",
        margin=dict(l=40, r=40, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        height=height,
    )


def create_tradingview_candlestick(df: pd.DataFrame) -> go.Figure:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Market Data Available", template="plotly_dark")
        return fig

    df_plot = df.copy()
    if len(df_plot) >= 20:
        df_plot["ema_20"] = df_plot["close"].ewm(span=20, adjust=False).mean()
    if len(df_plot) >= 50:
        df_plot["ema_50"] = df_plot["close"].ewm(span=50, adjust=False).mean()

    vol_colors = ["#00C087" if c >= o else "#FF3B69" for c, o in zip(df_plot["close"].fillna(0.0), df_plot.get("open", df_plot["close"]).fillna(0.0))]
    x_axis = df_plot["timestamp"] if "timestamp" in df_plot.columns else df_plot.index

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=x_axis, open=df_plot.get("open", df_plot["close"]), high=df_plot.get("high", df_plot["close"]),
        low=df_plot.get("low", df_plot["close"]), close=df_plot["close"], name="BTC/USDT",
        increasing_line_color="#00C087", decreasing_line_color="#FF3B69",
        increasing_fillcolor="#00C087", decreasing_fillcolor="#FF3B69",
    ), row=1, col=1)

    if "ema_20" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["ema_20"], mode="lines", name="EMA 20", line=dict(color="#2962FF", width=1.5)), row=1, col=1)
    if "ema_50" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["ema_50"], mode="lines", name="EMA 50", line=dict(color="#FF9800", width=1.5)), row=1, col=1)
    if "volume" in df_plot.columns:
        fig.add_trace(go.Bar(x=x_axis, y=df_plot["volume"], name="Volume (BTC)", marker=dict(color=vol_colors, opacity=0.7)), row=2, col=1)

    fig.update_layout(**_base_layout("Bitcoin (BTC/USDT) Hourly Price Action & Trading Volume", height=520), xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, gridcolor="#222733")
    fig.update_yaxes(title_text="Volume (BTC)", row=2, col=1, gridcolor="#222733")
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1, gridcolor="#222733")
    return fig


def create_alpha_equity_curve(df: pd.DataFrame) -> go.Figure:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "total_equity_usd" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Portfolio Data Available", template="plotly_dark")
        return fig

    df_plot = df.copy()
    x_axis = df_plot["timestamp"] if "timestamp" in df_plot.columns else df_plot.index
    fig = go.Figure()
    fig.add_hline(y=float(INITIAL_CAPITAL_USD), line_dash="dot", line_color="#666666", annotation_text=f"Initial Capital (${INITIAL_CAPITAL_USD:,.2f})", annotation_position="bottom right")

    if "benchmark_btc_equity" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["benchmark_btc_equity"], mode="lines", name="Bitcoin Buy & Hold Benchmark ($)", line=dict(color="#F7931A", width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot["total_equity_usd"], mode="lines+markers", name="Mudra Strategy Equity ($)", line=dict(color="#00F2FE", width=2.8), marker=dict(size=5, color="#00C087")))

    fig.update_layout(**_base_layout("Paper-Trading Strategy Equity vs. Buy & Hold Benchmark ($10,000 Base)", height=380), hovermode="x unified")
    fig.update_yaxes(title_text="Portfolio Value (USD)", gridcolor="#222733")
    fig.update_xaxes(title_text="Time (UTC)", gridcolor="#222733")
    return fig


def create_reconciliation_scatter(df: pd.DataFrame) -> go.Figure:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "actual_price_at_t_plus_1" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Reconciliation Data Available", template="plotly_dark")
        return fig

    df_plot = df.copy()
    if "timestamp" in df_plot.columns:
        df_plot = df_plot.sort_values("timestamp")
    x_axis = df_plot["timestamp"] if "timestamp" in df_plot.columns else df_plot.index

    colors = ["#00C087" if c else "#FF3B69" for c in df_plot.get("direction_correct", [True] * len(df_plot))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot["actual_price_at_t_plus_1"], mode="lines", name="Actual Realized Market Close", line=dict(color="#FFFFFF", width=1.5)))

    pred_series = df_plot.get("predicted_price", pd.Series(0.0, index=df_plot.index)).fillna(0.0)
    actual_series = df_plot["actual_price_at_t_plus_1"].fillna(0.0)
    error_series = df_plot.get("absolute_error_usd", pd.Series(0.0, index=df_plot.index)).fillna(0.0)
    tag_series = df_plot.get("error_attribution_tag", pd.Series("UNSETTLED", index=df_plot.index)).fillna("UNSETTLED")
    time_labels = [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M UTC") if pd.notnull(t) else str(t) for t in x_axis]
    is_correct_list = df_plot.get("direction_correct", [True] * len(df_plot))

    fig.add_trace(go.Scatter(
        x=x_axis, y=pred_series, mode="markers", name="Forecast Target",
        marker=dict(size=9, color=colors, line=dict(width=1, color="#FFFFFF")),
        text=[f"Time: {t}<br>Forecast: ${p:,.2f}<br>Actual: ${a:,.2f}<br>Error: ${e:,.2f}<br>Tag: {tag}<br>Result: {'HIT' if c else 'MISS'}"
              for t, p, a, e, tag, c in zip(time_labels, pred_series, actual_series, error_series, tag_series, is_correct_list)],
        hoverinfo="text",
    ))

    fig.update_layout(**_base_layout("Delayed Forecast Reconciliation (Market Close vs. Predicted Target)", height=380), hovermode="closest")
    fig.update_yaxes(title_text="Price (USD)", gridcolor="#222733")
    fig.update_xaxes(title_text="Time (UTC)", gridcolor="#222733")
    return fig
