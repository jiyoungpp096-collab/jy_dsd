# -*- coding: utf-8 -*-
"""
월별 매출 대시보드 (Streamlit, 빈 데이터/파싱 오류 안전 처리)
- 파일 업로드 또는 샘플 데이터 기반
- 6가지 시각화: KPI, 월별 추이(3M MA), 전년 대비 증감률, 누적 비교, 올해 vs 전년(이중축), 시즌성 히트맵, 월별 비중(도넛)

설치:
    pip install --upgrade streamlit pandas plotly kaleido
실행:
    streamlit run streamlit_app.py
"""
import io
import textwrap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="월별 매출 대시보드", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
def _fmt_won(v) -> str:
    try:
        return f"₩{int(v):,}"
    except Exception:
        return "-"

@st.cache_data
def load_sample() -> pd.DataFrame:
    csv = textwrap.dedent(
        """
        월,매출액,전년동월,증감률
        2024-01,12000000,10500000,14.3
        2024-02,13500000,11200000,20.5
        2024-03,11000000,12800000,-14.1
        2024-04,18000000,15200000,18.4
        2024-05,21000000,18500000,13.5
        2024-06,19000000,17500000,8.6
        2024-07,22000000,19800000,11.1
        2024-08,25000000,21000000,19.0
        2024-09,24000000,20500000,17.1
        2024-10,28000000,25000000,12.0
        2024-11,23000000,19500000,17.9
        2024-12,26000000,22000000,18.2
        """
    ).strip()
    return pd.read_csv(io.StringIO(csv))

@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 컬럼명 표준화
    out.columns = [c.strip() for c in out.columns]
    # 날짜 파싱 (유연 모드)
    out["월"] = pd.to_datetime(out["월"], errors="coerce")
    # 숫자 강제 변환
    for col in ["매출액", "전년동월", "증감률"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # 유효행만 남기기
    out = out.dropna(subset=["월", "매출액", "전년동월", "증감률"]).copy()
    if out.empty:
        return out
    out["연도"] = out["월"].dt.year
    out["월_숫자"] = out["월"].dt.month
    out = out.sort_values(["연도", "월"]).reset_index(drop=True)
    # 연도별 3개월 이동평균 (빈 그룹 대비 안전)
    out["3M_MA"] = (
        out.groupby("연도")["매출액"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    )
    # 누적
    out["누적매출"] = out.groupby("연도")["매출액"].cumsum()
    out["누적전년"] = out.groupby("연도")["전년동월"].cumsum()
    return out

# ------------------------------
# Sidebar - Data input & filter
# ------------------------------
st.sidebar.header("데이터 입력")
file = st.sidebar.file_uploader("CSV 업로드 (열: 월, 매출액, 전년동월, 증감률)", type=["csv"])
use_sample = st.sidebar.toggle("샘플 데이터 사용", value=True if not file else False)

if file:
    try:
        df_raw = pd.read_csv(file)
    except Exception as e:
        st.error(f"CSV 로드 오류: {e}")
        st.stop()
elif use_sample:
    df_raw = load_sample()
else:
    st.stop()

required_cols = {"월", "매출액", "전년동월", "증감률"}
if not required_cols.issubset(df_raw.columns):
    st.error(f"필수 컬럼 {required_cols} 이(가) 모두 존재해야 합니다. 현재: {list(df_raw.columns)}")
    st.stop()

_df = preprocess(df_raw)
if _df.empty:
    st.warning("유효한 행이 없습니다. 날짜/숫자 파싱을 확인해 주세요.")
    st.dataframe(df_raw)
    st.stop()

years = sorted(_df["연도"].dropna().unique().tolist())
# 옵션 구성과 기본 선택(최근 연도)
options = ["전체"] + years
default_index = 0 if not years else len(options) - 1
selected_year = st.sidebar.selectbox("연도 필터", options=options, index=default_index)

if selected_year == "전체":
    df_view = _df.copy()
    recent_year = (max(years) if years else None)
    df_recent = _df[_df["연도"] == recent_year].copy() if recent_year else _df.copy()
else:
    df_view = _df[_df["연도"] == selected_year].copy()
    df_recent = df_view.copy()

# ------------------------------
# Header
# ------------------------------
st.title("월별 매출 대시보드")
st.caption("CSV 업로드 → 연도 선택 → 6가지 시각화 자동 생성")

# ------------------------------
# KPI Cards
# ------------------------------
def render_kpis(df_show: pd.DataFrame):
    if df_show is None or df_show.empty:
        c1, c2, c3, c4 = st.columns(4)
        for c in (c1, c2, c3, c4):
            c.metric("-", "-", "")
        return
    total = float(df_show["매출액"].sum())
    total_prev = float(df_show["전년동월"].sum())
    delta = ((total - total_prev) / total_prev * 100) if total_prev else 0.0
    # 안전 idx 사용
    max_idx = df_show["매출액"].idxmax()
    min_idx = df_show["매출액"].idxmin()
    max_row = df_show.loc[max_idx]
    min_row = df_show.loc[min_idx]
    max_gr = df_show.loc[df_show["증감률"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 매출액", _fmt_won(total), f"{delta:+.1f}%")
    c2.metric("최대 매출 월", max_row["월"].strftime("%Y-%m"), _fmt_won(max_row["매출액"]))
    c3.metric("최소 매출 월", min_row["월"].strftime("%Y-%m"), _fmt_won(min_row["매출액"]))
    c4.metric("최대 증감률", max_gr["월"].strftime("%Y-%m"), f"{max_gr['증감률']:.1f}%")

render_kpis(df_recent)

# ------------------------------
# Chart builders (빈 데이터 대비 안전 처리)
# ------------------------------
def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, annotations=[dict(text="데이터 없음", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)])
    return fig


def chart_line_ma(df_show: pd.DataFrame) -> go.Figure:
    if df_show is None or df_show.empty:
        return _empty_fig("① 월별 매출 추이 (3M 이동평균)")
    df_show = df_show.sort_values("월")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_show["월"], y=df_show["매출액"], mode="lines+markers", name="매출액"))
    fig.add_trace(go.Scatter(x=df_show["월"], y=df_show["3M_MA"], mode="lines", name="3M 이동평균", line=dict(dash="dot")))
    # annotate (가드)
    try:
        max_row = df_show.loc[df_show["매출액"].idxmax()]
        min_row = df_show.loc[df_show["매출액"].idxmin()]
        fig.add_trace(go.Scatter(x=[max_row["월"]], y=[max_row["매출액"]], mode="markers+text", name="최대",
                                 text=[f"최대 {_fmt_won(max_row['매출액'])}"], textposition="top center"))
        fig.add_trace(go.Scatter(x=[min_row["월"]], y=[min_row["매출액"]], mode="markers+text", name="최소",
                                 text=[f"최소 {_fmt_won(min_row['매출액'])}"], textposition="bottom center"))
    except Exception:
        pass
    fig.update_layout(title="① 월별 매출 추이 (3M 이동평균)", xaxis_title="월", yaxis_title="매출액", legend=dict(orientation="h"))
    return fig


def chart_yoy_bar(df_show: pd.DataFrame) -> go.Figure:
    if df_show is None or df_show.empty:
        return _empty_fig("② 전년 대비 증감률(%)")
    df_show = df_show.sort_values("월")
    colors = ["#58d68d" if (pd.notna(v) and v >= 0) else "#ff6b6b" for v in df_show["증감률"].values]
    fig = go.Figure(go.Bar(x=df_show["월"], y=df_show["증감률"], marker_color=colors, name="증감률(%)"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="② 전년 대비 증감률(%)", xaxis_title="월", yaxis_title="증감률(%)", legend=dict(orientation="h"))
    return fig


def chart_cumulative(df_show: pd.DataFrame) -> go.Figure:
    if df_show is None or df_show.empty:
        return _empty_fig("③ 누적 매출 비교")
    df_show = df_show.sort_values("월").copy()
    df_show["누적매출_tmp"] = df_show["매출액"].cumsum()
    df_show["누적전년_tmp"] = df_show["전년동월"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_show["월"], y=df_show["누적매출_tmp"], mode="lines", name="올해 누적"))
    fig.add_trace(go.Scatter(x=df_show["월"], y=df_show["누적전년_tmp"], mode="lines", name="전년 누적", fill="tonexty"))
    fig.update_layout(title="③ 누적 매출 비교", xaxis_title="월", yaxis_title="누적 매출액", legend=dict(orientation="h"))
    return fig


def chart_dual_axis(df_show: pd.DataFrame) -> go.Figure:
    if df_show is None or df_show.empty:
        return _empty_fig("④ 올해 vs 전년 월별 비교 (이중축)")
    df_show = df_show.sort_values("월")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_show["월"], y=df_show["매출액"], name="올해 매출", opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_show["월"], y=df_show["전년동월"], mode="lines+markers", name="전년 동월"), secondary_y=True)
    fig.update_yaxes(title_text="올해(원)", secondary_y=False)
    fig.update_yaxes(title_text="전년(원)", secondary_y=True)
    fig.update_layout(title="④ 올해 vs 전년 월별 비교 (이중축)", xaxis_title="월", legend=dict(orientation="h"))
    return fig


def chart_season_heatmap(df_all: pd.DataFrame) -> go.Figure:
    if df_all is None or df_all.empty:
        return _empty_fig("⑤ 시즌성 히트맵")
    years = sorted(df_all["연도"].dropna().unique().tolist())
    if not years:
        return _empty_fig("⑤ 시즌성 히트맵")
    pivot = df_all.pivot_table(index="연도", columns="월_숫자", values="매출액", aggfunc="sum")
    # 열/행이 비어도 안전하게 재정렬
    pivot = pivot.reindex(index=years)
    pivot = pivot.reindex(columns=range(1, 13))
    if pivot.size == 0:
        return _empty_fig("⑤ 시즌성 히트맵")
    fig = px.imshow(
        pivot,
        labels=dict(x="월", y="연도", color="매출액"),
        x=[f"{m:02d}월" for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(title="⑤ 시즌성 히트맵")
    return fig


def chart_share_pie(df_show: pd.DataFrame) -> go.Figure:
    if df_show is None or df_show.empty:
        return _empty_fig("⑥ 월별 매출 비중 (도넛)")
    fig = go.Figure(go.Pie(labels=df_show["월"].dt.strftime("%Y-%m"), values=df_show["매출액"], hole=0.5))
    fig.update_layout(title="⑥ 월별 매출 비중 (도넛)", legend=dict(orientation="h"))
    return fig

# ------------------------------
# Layout & Render
# ------------------------------
col1, col2 = st.columns(2)
col1.plotly_chart(chart_line_ma(df_recent), use_container_width=True)
col2.plotly_chart(chart_yoy_bar(df_recent), use_container_width=True)

col3, col4 = st.columns(2)
col3.plotly_chart(chart_cumulative(df_recent), use_container_width=True)
col4.plotly_chart(chart_dual_axis(df_recent), use_container_width=True)

col5, col6 = st.columns(2)
col5.plotly_chart(chart_season_heatmap(_df), use_container_width=True)
col6.plotly_chart(chart_share_pie(df_recent), use_container_width=True)

# ------------------------------
# Downloads (optional)
# ------------------------------
st.divider()
st.subheader("다운로드")

def _png_download_button(fig: go.Figure, label: str, key: str):
    try:
        img_bytes = fig.to_image(format="png", width=1280, height=720, scale=2)
        st.download_button(label, data=img_bytes, file_name=f"{key}.png", mime="image/png")
    except Exception:
        st.info("PNG 저장에는 kaleido가 필요합니다. 설치 후 다시 시도하세요.")

with st.expander("차트 PNG로 저장 (선택)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        _png_download_button(chart_line_ma(df_recent), "① 월별 추이(PNG)", "chart1")
        _png_download_button(chart_cumulative(df_recent), "③ 누적 비교(PNG)", "chart3")
    with c2:
        _png_download_button(chart_yoy_bar(df_recent), "② 증감률(PNG)", "chart2")
        _png_download_button(chart_dual_axis(df_recent), "④ 이중축(PNG)", "chart4")
    with c3:
        _png_download_button(chart_season_heatmap(_df), "⑤ 히트맵(PNG)", "chart5")
        _png_download_button(chart_share_pie(df_recent), "⑥ 도넛(PNG)", "chart6")

st.caption("ⓘ 연도 필터는 사이드바에서 변경할 수 있으며, 모든 차트가 동기화됩니다.")

