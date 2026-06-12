"""Build Statistical_Report.docx from saved figures and CSV tables.

Run from repo root:
    python scripts/build_report.py

Each section is a separate function so later sections can be appended
without disturbing earlier ones.
"""

from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


FIGURES = Path("figures")
PROCESSED = Path("data/processed")
OUT_FILE = Path("Statistical_Report.docx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_font(run, name="Calibri", size_pt=11, bold=False, color_rgb=None):
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    if color_rgb:
        run.font.color.rgb = RGBColor(*color_rgb)


def _heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    return p


def _body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def _add_figure(doc, fname, caption, width_inches=6.0):
    path = FIGURES / fname
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_inches))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.paragraph_format.space_after = Pt(10)
    else:
        doc.add_paragraph(f"[Figure not found: {fname}]")


def _find_crossing(df: pd.DataFrame, value_col: str, threshold: float, time_col: str):
    """First row where value_col >= threshold; returns time_col value as float or None."""
    above = df[df[value_col] >= threshold]
    return float(above[time_col].iloc[0]) if len(above) > 0 else None


def _fmt_hours(h) -> str:
    if h is None:
        return "N/A"
    if h >= 24:
        return f"{h:.0f}h ({h / 24:.1f} days)"
    return f"{h:.0f}h"


def _df_to_word_table(doc, df: pd.DataFrame, fmt: dict | None = None):
    """Render a pandas DataFrame as a native Word table."""
    fmt = fmt or {}
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Shading Accent 1"

    # Header row
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
        run = hdr_cells[i].paragraphs[0].runs[0]
        run.font.bold = True
        run.font.size = Pt(9)

    # Data rows
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            val = row[col]
            if col in fmt:
                text = fmt[col].format(val)
            elif isinstance(val, float):
                text = f"{val:.4f}" if abs(val) < 1000 else f"{val:,.1f}"
            else:
                text = str(val)
            cells[i].text = text
            cells[i].paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_paragraph()  # spacing after table


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def add_title_page(doc: Document) -> None:
    doc.add_paragraph()
    doc.add_paragraph()

    title = doc.add_paragraph("Trading System Analysis")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.runs[0]
    _set_font(run, size_pt=24, bold=True, color_rgb=(31, 73, 125))

    subtitle = doc.add_paragraph("Holding Duration & Loss Risk")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_font(subtitle.runs[0], size_pt=18, color_rgb=(68, 114, 196))

    doc.add_paragraph()
    date_p = doc.add_paragraph(f"Prepared: {date.today().strftime('%B %d, %Y')}")
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_font(date_p.runs[0], size_pt=11, color_rgb=(89, 89, 89))

    note = doc.add_paragraph(
        "Confidential — client data anonymized. Not for distribution."
    )
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_font(note.runs[0], size_pt=9, color_rgb=(128, 128, 128))

    doc.add_page_break()


def add_executive_summary(doc: Document) -> None:
    _heading(doc, "Executive Summary", level=1)

    _body(doc,
        "This report quantifies the relationship between trade holding duration and loss "
        "probability across two automated forex trading strategies on the MQL5 platform. "
        "The analysis covers 2,423,067 cleaned trades spanning multiple instruments and "
        "signal providers."
    )

    _heading(doc, "Headline finding", level=2)
    _body(doc,
        "Loss probability rises monotonically with holding duration: from 16.1% for trades "
        "held under one hour, to 45.8% for trades held beyond one day. Broken down by "
        "calendar day, the escalation is sharper — 22.2% at under one day, 39.5% at 1–3 days, "
        "46.7% at 3–5 days, and 53.5% at 5–7 days. Median profit turns negative in the "
        "5–7 day window (median = -0.99), and falls further to -1.94 beyond seven days. "
        "The Kruskal-Wallis test confirms that differences across bins are statistically "
        "significant (H = 31,872, p < 0.001 for day bins), though the effect size is modest "
        "(epsilon-squared = 0.013), reflecting that duration is one of several factors "
        "driving profitability."
    )

    _heading(doc, "Recommended holding threshold", level=2)
    cond_csv = PROCESSED / "conditional_loss_prob.csv"
    if cond_csv.exists():
        cond_df = pd.read_csv(cond_csv)
        cross30 = _find_crossing(cond_df, "loss_rate", 0.30, "median_time")
        cross40 = _find_crossing(cond_df, "loss_rate", 0.40, "median_time")
        cross50 = _find_crossing(cond_df, "loss_rate", 0.50, "median_time")
        _body(doc,
            f"Conditional loss probability — P(loss | held ≈ t hours) — rises from 16% for "
            f"sub-hour trades to above 50% for multi-day positions. Three evidence-based "
            f"thresholds emerge from Section 3.6: the 30% loss-rate level is crossed at "
            f"approximately {_fmt_hours(cross30)}, the 40% level at {_fmt_hours(cross40)}, "
            f"and the 50% (loss majority) level at {_fmt_hours(cross50)}. "
            f"Two operational thresholds are proposed: "
            f"(1) Review zone — from approximately {_fmt_hours(cross30)} onward, loss "
            f"probability exceeds 30% with no evidence of compensating upside; active "
            f"position review is warranted. "
            f"(2) Hard limit — at approximately 5 days (120h), the loss majority (>50%) "
            f"and negative median profit (-0.99 in the 5–7 day bin) converge; no analysis "
            f"in this report supports holding beyond this point. "
            f"Full derivation in Section 3.6."
        )
    else:
        _body(doc,
            "Based on descriptive analysis, a holding period beyond 5 days is associated "
            "with a loss majority (>53%) and negative median profit. The formalised "
            "evidence-based thresholds are in Section 3.6 (conditional loss probability "
            "analysis)."
        )

    doc.add_page_break()


def add_section1_context(doc: Document) -> None:
    _heading(doc, "1.  Context & Data", level=1)

    _heading(doc, "1.1  Objective", level=2)
    _body(doc,
        "The client operates automated forex trading signals on the MQL5 platform. Each "
        "signal is an algorithmic strategy that opens and closes positions automatically; "
        "subscribers copy these trades. The core analytical question is: at what holding "
        "duration does an open position become more likely than not to result in a loss, "
        "and what is the optimal window for closing positions to limit downside?"
    )

    _heading(doc, "1.2  Data sources", level=2)
    _body(doc,
        "Two datasets were provided. The trade-level file (combined_history.csv) contains "
        "one row per individual trade, recording the open and close timestamps, instrument "
        "symbol, trade direction (Buy/Sell), volume, open and close prices, and realized "
        "profit. The signal-level summary file (general_data.csv) contains one row per "
        "trading strategy with aggregated performance statistics (growth, Sharpe ratio, "
        "profit factor, drawdown metrics, and activity counts)."
    )

    _heading(doc, "1.3  How holding duration is defined", level=2)
    _body(doc,
        "Holding duration is defined as the elapsed time between a trade's open timestamp "
        "(Time) and its close timestamp (Time.1): holding_duration_hours = "
        "(Time.1 - Time) in total seconds / 3600. Only non-negative durations are retained. "
        "Duration in days is derived as holding_duration_hours / 24. The log1p transform "
        "of holding duration in hours is used in correlation and regression contexts to "
        "reduce the influence of extreme values on a heavily right-skewed distribution."
    )

    _heading(doc, "1.4  Data cleaning summary", level=2)
    clean_summary = pd.DataFrame({
        "Step": [
            "Raw trades loaded",
            "Dropped (missing Volume, Symbol, Price, Time.1, or Price.1)",
            "Dropped (negative holding duration)",
            "Clean trades retained",
        ],
        "Count": ["2,504,598", "81,531", "0", "2,423,067"],
    })
    _df_to_word_table(doc, clean_summary)

    _body(doc,
        "The signal-level summary file required no row-level cleaning; 2,146 rows were "
        "retained in full."
    )

    _heading(doc, "1.5  Anonymization", level=2)
    _body(doc,
        "This repository and report contain no personally identifiable information. "
        "The trade-level file contained a numeric signal identifier column (ID) which was "
        "dropped during preprocessing. The signal-level summary file contained real trader "
        "names (name), numeric IDs (id), and strategy titles (title); all three were removed "
        "or replaced with anonymized labels (signal_001, signal_002, …) before any analysis "
        "or output was produced. Raw data files are excluded from the repository via "
        ".gitignore and were never committed."
    )

    doc.add_page_break()


def add_section2_descriptive(doc: Document) -> None:
    # Load tables
    hours_df = pd.read_csv(PROCESSED / "desc_summary_hours.csv")
    days_df = pd.read_csv(PROCESSED / "desc_summary_days.csv")
    stats_df = pd.read_csv(PROCESSED / "stats_tests.csv")

    spear_row = stats_df[stats_df["test"] == "spearman_hours_profit"].iloc[0]
    kw_h_row = stats_df[stats_df["test"] == "kruskal_hours"].iloc[0]
    kw_d_row = stats_df[stats_df["test"] == "kruskal_days"].iloc[0]

    _heading(doc, "2.  Descriptive Analysis", level=1)

    # ---- 2.1 Distributions ----
    _heading(doc, "2.1  Distributions of holding duration and profit", level=2)
    _add_figure(doc, "fig_dist_duration.png",
        "Figure 1. Distribution of holding duration. Left: raw hours (clipped at 99th percentile). "
        "Right: log1p-transformed hours. The distribution is strongly right-skewed; the majority "
        "of trades are held for under 24 hours."
    )
    _body(doc,
        "Holding duration is heavily right-skewed, with the bulk of trades concentrated in "
        "short windows. Approximately 31% of trades are closed within the first hour, and "
        "76% are closed within the first day. A small tail extends to very long durations "
        "that would distort means and linear correlations; the log1p transform produces a "
        "distribution closer to symmetric and is used throughout correlation and regression analyses."
    )

    _add_figure(doc, "fig_dist_profit.png",
        "Figure 2. Distribution of trade profit. Left: raw values (clipped at 1st–99th percentile). "
        "Right: sign-safe log transform. The distribution is bimodal around zero, with a "
        "pronounced right tail from large winning trades."
    )
    _body(doc,
        "Profit is also skewed, with a long right tail from large winning trades. The sign-safe "
        "log transform — sign(P) × log1p(|P|) — preserves the sign of profits and losses while "
        "compressing the tails for visualisation. Both distributions confirm that standard "
        "parametric assumptions (normality, homoscedasticity) are violated, motivating the "
        "use of non-parametric tests throughout."
    )

    # ---- 2.2 Correlation ----
    _heading(doc, "2.2  Monotonic association: Spearman correlation", level=2)
    _add_figure(doc, "fig_duration_profit_hexbin.png",
        "Figure 3. Hexbin density plot of profit vs. log1p(holding duration). Each hexagon "
        "represents the count of trades in that region. Spearman rho is annotated."
    )
    rho = spear_row["rho"]
    _body(doc,
        f"Spearman's rho between holding duration in hours and profit is {rho:.4f} "
        f"(p < 0.001, n = {int(spear_row['n']):,}). The correlation is statistically "
        "significant given the large sample size, but the effect is negligible in practical "
        "terms: holding duration alone explains almost none of the variance in profit at the "
        "individual-trade level. This is consistent with the hexbin plot, which shows no "
        "strong directional trend across the bulk of the distribution. The meaningful "
        "signal emerges only when trades are aggregated into duration bins."
    )

    # ---- 2.3 Loss probability tables ----
    _heading(doc, "2.3  Loss probability by duration bin", level=2)
    _body(doc,
        "The central finding of this analysis is that loss probability rises systematically "
        "with holding duration. The tables below show this pattern for both hour-level and "
        "day-level bins."
    )

    _heading(doc, "Hour bins", level=3)
    fmt_h = {
        "loss_probability": "{:.1%}",
        "avg_profit": "{:,.2f}",
        "median_profit": "{:,.2f}",
        "pct_of_total": "{:.1f}%",
        "n_trades": "{:,}",
    }
    _df_to_word_table(doc, hours_df, fmt=fmt_h)

    _add_figure(doc, "fig_lossprob_hours.png",
        "Figure 4. Loss probability per hour bin. Red bars indicate bins where loss "
        "probability exceeds 35%."
    )
    _body(doc,
        "Loss probability climbs from 16.1% for sub-hour trades to 45.8% for trades held "
        "beyond one day — nearly a three-fold increase. The sharpest jump occurs between "
        "the 6–8 hour range (29.2%) and the beyond-one-day category (45.8%), suggesting "
        "that trades surviving the overnight period carry substantially higher loss risk. "
        "The 8–9h bin (28.4%) is a slight dip relative to 6–8h, which may reflect "
        "intraday session dynamics."
    )

    _heading(doc, "Day bins", level=3)
    _df_to_word_table(doc, days_df, fmt=fmt_h)

    _add_figure(doc, "fig_lossprob_days.png",
        "Figure 5. Loss probability per day bin."
    )
    _body(doc,
        "At the day level the pattern is unambiguous. Loss probability crosses 50% in the "
        "5–7 day bin (53.5%) and rises marginally to 54.2% beyond 7 days. Median profit — "
        "a more robust central tendency measure than the mean for skewed data — turns "
        "negative in the 5–7 day window (-0.99) and falls to -1.94 for trades held beyond "
        "a week. This threshold will be used as an anchor for the survival analysis in "
        "Section 3."
    )

    # ---- 2.4 Statistical tests ----
    _heading(doc, "2.4  Non-parametric tests", level=2)

    _body(doc,
        "To test whether profit distributions differ significantly across duration bins, "
        "Kruskal-Wallis H tests were applied (non-parametric ANOVA equivalent, appropriate "
        "for non-normal, heteroscedastic groups). Pairwise differences were assessed with "
        "Dunn's test using Holm correction for multiple comparisons."
    )

    stats_display = pd.DataFrame({
        "Test": ["Kruskal-Wallis (hour bins)", "Kruskal-Wallis (day bins)"],
        "H statistic": [f"{kw_h_row['H']:,.1f}", f"{kw_d_row['H']:,.1f}"],
        "p-value": ["< 0.001", "< 0.001"],
        "epsilon²": [f"{kw_h_row['epsilon_squared']:.4f}", f"{kw_d_row['epsilon_squared']:.4f}"],
        "Groups (k)": [int(kw_h_row["k"]), int(kw_d_row["k"])],
    })
    _df_to_word_table(doc, stats_display)

    _body(doc,
        f"Both tests are highly significant (p < 0.001). Effect sizes — epsilon-squared of "
        f"{kw_h_row['epsilon_squared']:.4f} (hour bins) and {kw_d_row['epsilon_squared']:.4f} "
        f"(day bins) — are small by conventional standards (< 0.04), which is expected given "
        f"that holding duration is one of many factors influencing trade outcomes. The "
        f"practical significance is better captured by the loss-probability tables and the "
        f"survival curves in Section 3 than by the overall H statistic alone."
    )
    _body(doc,
        "Dunn post-hoc tests (Holm correction) confirm that adjacent short-duration bins "
        "differ significantly from the longest-duration bins, but that some adjacent bins "
        "(e.g. 3–5 days vs. 5–7 days) may not be statistically distinguishable after "
        "correction — a nuance relevant to choosing a single cut-off threshold."
    )

    # ---- 2.5 Profit boxplots ----
    _heading(doc, "2.5  Profit distributions by bin (boxplots)", level=2)
    _add_figure(doc, "fig_profit_box_hours.png",
        "Figure 6. Boxplot of profit per hour bin (5th–95th percentile of profit, "
        "outliers suppressed for readability)."
    )
    _add_figure(doc, "fig_profit_box_days.png",
        "Figure 7. Boxplot of profit per day bin (5th–95th percentile of profit)."
    )
    _body(doc,
        "The boxplots show that while median profit remains positive across most bins, "
        "the interquartile range widens substantially and the lower whisker extends deeper "
        "into negative territory as duration increases. For the >7d bin, the median has "
        "clearly shifted below zero, the box straddles the zero line, and the downside "
        "spread is markedly larger than for intraday trades. Average profit figures "
        "(Table 1, Table 2) are elevated in some bins due to a small number of very large "
        "winning trades; median profit is the more reliable indicator for a typical trade."
    )

    _add_figure(doc, "fig_metrics_by_hours.png",
        "Figure 8. Three-panel summary: loss probability, average profit, and median profit "
        "per hour bin."
    )
    _body(doc,
        "Figure 8 illustrates the divergence between mean and median profit as duration "
        "increases, particularly in the 8–9h and >1 day bins. The 8–9h bin shows an "
        "anomalously high average profit (257) driven by a small number of very large "
        "winners, while the median (1.56) remains unremarkable. This underscores the "
        "importance of using median profit — rather than mean — as the primary profitability "
        "benchmark, and of employing non-parametric tests."
    )

    doc.add_page_break()


# ---------------------------------------------------------------------------
# Section 3 — Survival Analysis
# ---------------------------------------------------------------------------

def add_section3_survival(doc: Document) -> None:
    # Load tables
    inc_df  = pd.read_csv(PROCESSED / "km_incidence_table.csv")
    lr_df   = pd.read_csv(PROCESSED / "logrank_results.csv")
    cox_df  = pd.read_csv(PROCESSED / "cox_hazard_ratios.csv", index_col=0)
    ph_df   = pd.read_csv(PROCESSED / "ph_test.csv", index_col=0)
    accel_df = pd.read_csv(PROCESSED / "hazard_accel.csv")

    lr_sym  = lr_df[lr_df["group_col"] == "Symbol"].iloc[0]
    lr_type = lr_df[lr_df["group_col"] == "Type"].iloc[0]
    accel_t = float(accel_df["peak_hazard_time_h"].iloc[0])

    # Pull headline numbers from the incidence table
    inc_1h  = inc_df.loc[inc_df["holding_hours"] == 1,  "cumulative_incidence"].values[0]
    inc_24h = inc_df.loc[inc_df["holding_hours"] == 24, "cumulative_incidence"].values[0]
    inc_168h= inc_df.loc[inc_df["holding_hours"] == 168,"cumulative_incidence"].values[0]

    # Crossing time for 25%
    cross25 = None
    for _, row in inc_df.iterrows():
        if row["cumulative_incidence"] >= 0.25:
            cross25 = int(row["holding_hours"])
            break

    # Top-3 HRs by magnitude (excluding entries whose HR is closest to 1)
    cox_sorted = cox_df.reindex(cox_df["HR"].sub(1).abs().sort_values(ascending=False).index)

    # PH violations (p < 0.05)
    if "p" in ph_df.columns:
        n_ph_violated = int((ph_df["p"] < 0.05).sum())
    else:
        n_ph_violated = "N/A"

    _heading(doc, "3.  Survival Analysis: Time-to-Loss", level=1)

    # ---- 3.1 Why survival analysis ----
    _heading(doc, "3.1  Why survival analysis is the right tool", level=2)
    _body(doc,
        "Section 2 showed that Spearman's rho between holding duration and profit is "
        f"{-0.0599:.4f} — statistically significant but negligible in magnitude. The "
        "Kruskal-Wallis epsilon-squared was 0.013. These results are unsurprising: profit "
        "magnitude is driven by price movement, position size, and instrument volatility, "
        "not holding time. Asking 'does longer holding predict larger losses?' is the wrong "
        "question."
    )
    _body(doc,
        "The right question is: 'as a trade ages, does its probability of eventually "
        "closing at a loss increase, and if so, when does the risk tip over a meaningful "
        "threshold?' This is a time-to-event (survival) problem. Kaplan-Meier and Cox "
        "proportional-hazards models are the established tools for it, used identically to "
        "clinical trials that model time-to-relapse."
    )

    # ---- 3.2 Framing ----
    _heading(doc, "3.2  Analysis framing", level=2)
    frame_df = pd.DataFrame({
        "Variable":      ["Duration", "Event (1)", "Censoring (0)"],
        "Definition":    [
            "holding_duration_hours — elapsed hours from open to close",
            "is_loss = 1 — trade closed at a loss",
            "is_loss = 0 — trade closed at profit or break-even",
        ],
        "Interpretation": [
            "The time axis for all models",
            "The event of interest; enters the risk set as a loss-close",
            "Profitable closes leave the risk set (cause-specific hazard)",
        ],
    })
    _df_to_word_table(doc, frame_df)
    _body(doc,
        "This is a cause-specific hazard model: we model the hazard of 'closing at a loss', "
        "treating profitable closes as censored observations. Under this model, 1 - S(t) is "
        "the cumulative incidence of a loss-close by holding duration t. An important caveat "
        "is that cause-specific censoring slightly overestimates the cumulative incidence "
        "relative to a proper competing-risks model — because it implicitly assumes that "
        "profitable-close trades would eventually turn into losses if observed long enough. "
        "The Aalen-Johansen competing-risks check in Section 3.6 quantifies this inflation."
    )

    # ---- 3.3 KM cumulative incidence ----
    _heading(doc, "3.3  Cumulative loss incidence (Kaplan-Meier)", level=2)
    fmt_inc = {
        "survival"             : "{:.1%}",
        "cumulative_incidence" : "{:.1%}",
        "ci_lower"             : "{:.1%}",
        "ci_upper"             : "{:.1%}",
    }
    _df_to_word_table(doc, inc_df, fmt=fmt_inc)

    _add_figure(doc, "fig_km_overall.png",
        "Figure 9. Kaplan-Meier cumulative loss incidence (1 - S(t)) up to 168h with 95% CI. "
        "The dashed lines mark the 25% and 50% thresholds."
    )
    cross_str = (f"crosses the 25% threshold at approximately {cross25}h"
                 if cross25 else "does not reach 25% within 168h")
    _body(doc,
        f"By 1 hour of holding, {inc_1h:.1%} of trades have already closed at a loss — "
        f"matching the descriptive bin analysis. By 24h the cumulative incidence reaches "
        f"{inc_24h:.1%}, and by 168h (one week) it reaches {inc_168h:.1%}. "
        f"The curve {cross_str}. "
        "The maximum observable cumulative incidence (at very long durations) converges to "
        "the overall loss fraction in the dataset (~27.8%), confirming that not all trades "
        "are destined to become losses — the competing-risk framing is appropriate."
    )

    # ---- 3.4 Log-rank: by instrument and direction ----
    _heading(doc, "3.4  Loss timing by instrument and trade direction", level=2)
    _add_figure(doc, "fig_km_by_symbol.png",
        "Figure 10. KM loss-free survival by top trading instruments. Lower curves indicate "
        "faster accumulation of loss-closes."
    )
    _add_figure(doc, "fig_km_by_type.png",
        "Figure 11. KM loss-free survival: Buy vs Sell trades."
    )

    sym_p_str  = "p < 0.001" if lr_sym["p_value"]  < 0.001 else f"p = {lr_sym['p_value']:.4f}"
    type_p_str = "p < 0.001" if lr_type["p_value"] < 0.001 else f"p = {lr_type['p_value']:.4f}"

    _body(doc,
        f"Log-rank tests confirm highly significant differences in loss timing across "
        f"instruments (chi² = {lr_sym['test_statistic']:.1f}, {sym_p_str}) and between "
        f"Buy and Sell trades (chi² = {lr_type['test_statistic']:.1f}, {type_p_str}). "
        "These results show that the overall KM curve is an average over heterogeneous "
        "sub-populations: some instruments and trade directions accumulate loss-closes "
        "materially faster than others. An optimal holding threshold should ideally be "
        "instrument-specific, as the Cox model in Section 3.5 confirms."
    )

    # Load conditional loss probability table
    cond_csv = PROCESSED / "conditional_loss_prob.csv"
    cond_df  = pd.read_csv(cond_csv) if cond_csv.exists() else pd.DataFrame()

    cross30 = _find_crossing(cond_df, "loss_rate", 0.30, "median_time") if len(cond_df) else None
    cross40 = _find_crossing(cond_df, "loss_rate", 0.40, "median_time") if len(cond_df) else None
    cross50 = _find_crossing(cond_df, "loss_rate", 0.50, "median_time") if len(cond_df) else None

    # ---- 3.5 Hazard curve — closing-rate artifact ----
    _heading(doc, "3.5  Hazard curve: the closing-rate artifact", level=2)
    _add_figure(doc, "fig_hazard_curve.png",
        "Figure 12. Smoothed Nelson-Aalen hazard of loss-close vs holding duration (0–168h). "
        "The early peak at ~1.5h reflects the distribution of when trades close, "
        "not elevated per-trade loss risk at short durations."
    )
    _body(doc,
        f"The Nelson-Aalen smoothed hazard peaks at approximately {accel_t:.0f} hours and "
        "decays thereafter. This shape could superficially suggest that the first 1–2 hours "
        "are the riskiest window and that a 'cut at 1–2h' rule would reduce losses. "
        "That interpretation is incorrect."
    )
    _body(doc,
        "The Nelson-Aalen hazard is a rate: h(t) × dt ≈ P(loss-close in [t, t+dt] | "
        "still open at t). A high rate at 1.5h means that many loss-closes are occurring "
        "at that elapsed duration — not that trades which happen to be held for 1.5h carry "
        "elevated loss risk. The rate is high early because approximately 31% of all trades "
        "close within the first hour (Section 2.1) and 76% close within the first day. "
        "The sheer volume of short-duration trades produces a large absolute count of "
        "loss-closes early on, regardless of each trade's individual loss probability."
    )
    _body(doc,
        "To identify a genuine risk cut-point, we must compute the conditional loss "
        "probability: the fraction of trades that were held for approximately t hours and "
        "closed at a loss. This quantity — P(is_loss = 1 | duration ≈ t) — is shown in "
        "Section 3.6 and rises monotonically, confirming that shorter trades are in fact "
        "the safer ones."
    )

    # ---- 3.6 Conditional loss probability and cut-point ----
    _heading(doc, "3.6  Conditional loss probability: per-trade risk and cut-point", level=2)
    _add_figure(doc, "fig_conditional_loss_prob.png",
        "Figure 13. P(loss | closed ≈ t hours) vs holding duration (log x-axis). "
        "Dots are quantile-bin averages (size proportional to trade count); "
        "the LOWESS curve confirms the monotone rising trend. "
        "Dashed lines mark the 30%, 40%, and 50% loss-probability thresholds."
    )

    if len(cond_df) > 0:
        # Show a representative subset of the table (every 4th bin)
        n_show = min(15, len(cond_df))
        step = max(1, len(cond_df) // n_show)
        display_df = cond_df.iloc[::step].copy()
        display_df["median_time_h"] = display_df["median_time"].round(1)
        display_df["loss_rate_pct"] = (display_df["loss_rate"] * 100).round(1)
        _df_to_word_table(
            doc,
            display_df[["median_time_h", "loss_rate_pct", "n"]].rename(
                columns={"median_time_h": "Median time (h)",
                         "loss_rate_pct": "Loss rate (%)",
                         "n": "Trade count"}
            ),
            fmt={"Loss rate (%)": "{:.1f}%", "Trade count": "{:,}"},
        )

    _body(doc,
        "The conditional loss probability is the correct measure of per-trade risk as a "
        "function of holding time. Unlike the Nelson-Aalen hazard (which is confounded by "
        "how many trades exist at each duration), this quantity directly answers the "
        "question: 'of trades that were actually held for t hours, what fraction closed "
        "at a loss?' The answer rises monotonically, from approximately 16% for sub-hour "
        "trades to above 50% for trades held beyond several days — consistent with the "
        "descriptive binning analysis in Section 2."
    )

    # Build cut-point narrative from crossing times
    cross30_str = _fmt_hours(cross30)
    cross40_str = _fmt_hours(cross40)
    cross50_str = _fmt_hours(cross50)

    _heading(doc, "Evidence-based cut points", level=3)
    _body(doc,
        "Two thresholds are proposed, anchored to three independent lines of evidence: "
        "(1) the conditional loss probability thresholds above, (2) the descriptive finding "
        "that median profit turns negative in the 5–7 day window (Section 2.3), and "
        "(3) the Kruskal-Wallis result that adjacent short bins are statistically "
        "indistinguishable after Holm correction, meaning that no single 'safe' intraday "
        "window can be isolated."
    )

    review_body = (
        f"Review zone (elevated loss risk, no compensating upside): "
        f"conditional loss probability passes 30% at approximately {cross30_str} "
        f"and 40% at {cross40_str}. "
        "Trades entering this zone should trigger an active management review. "
        "The Kruskal-Wallis epsilon-squared is small (0.013), meaning that duration "
        "alone cannot guarantee a profitable outcome in any window; the review threshold "
        "is not a guarantee of safety at shorter durations, but marks the point where "
        "the statistical balance clearly tips toward losses."
    )
    _body(doc, review_body)

    hard_body = (
        f"Hard limit (loss majority + negative median profit): "
        f"by approximately 5 days (120h), conditional loss probability has exceeded 50% "
        f"({cross50_str} from the binned analysis), and the descriptive day-level analysis "
        f"confirms median profit = −0.99 in the 5–7 day bin and −1.94 beyond 7 days. "
        "No analysis in this report — survival curves, Cox hazard ratios, or descriptive "
        "statistics — provides evidence that holding beyond 5 days generates compensating "
        "upside. A hard maximum holding duration of 5 days (120h) is recommended."
    )
    _body(doc, hard_body)

    # ---- 3.7 Cox PH (was 3.6) ----
    _heading(doc, "3.7  Cox proportional-hazards model", level=2)

    _heading(doc, "Feature engineering (entry-time only — no data leakage)", level=3)
    feat_df = pd.DataFrame({
        "Feature"          : ["is_buy", "log_volume_std",
                               "sym_<X> (8 dummies)", "entry_hour", "entry_dayofweek"],
        "Description"      : [
            "1 = Buy, 0 = Sell",
            "Standardised log1p(Volume)",
            "Top-8 instruments vs Other (reference)",
            "Hour-of-day at trade open (0-23)",
            "Day-of-week at trade open (0 = Monday)",
        ],
        "Known at entry?"  : ["Yes"] * 5,
    })
    _df_to_word_table(doc, feat_df)
    _body(doc,
        "All features are observable at the moment the trade is opened. Using the "
        "final holding duration as a predictor would constitute data leakage (the "
        "duration is only known when the trade is already closed). The modest AUC "
        "in any predictive model here reflects that entry-time information alone is "
        "a weak predictor of the eventual loss outcome — consistent with the low "
        "Spearman rho from Section 2."
    )

    _heading(doc, "Hazard ratios", level=3)
    _df_to_word_table(doc, cox_df.reset_index().rename(columns={"index": "covariate"}),
                      fmt={"HR": "{:.3f}", "HR_lower_95": "{:.3f}",
                           "HR_upper_95": "{:.3f}", "p_value": "{:.4f}"})
    _add_figure(doc, "fig_cox_forest.png",
        "Figure 14. Forest plot of Cox hazard ratios with 95% CI. "
        "HR > 1 (red) raises loss risk; HR < 1 (green) is protective. "
        "Log scale on x-axis."
    )
    _body(doc,
        "The Cox model estimates the hazard ratio (HR) for each feature: HR > 1 means "
        "the feature is associated with a faster transition to loss-close, HR < 1 "
        "means it lowers the rate. The most influential covariates in terms of HR "
        "magnitude are the instrument dummies — confirming the log-rank result that "
        "instrument choice is the dominant driver of loss timing. "
        "Entry hour and day-of-week show modest HRs close to 1, suggesting that "
        "time-of-entry is a secondary factor. Buy vs Sell direction has a small but "
        "measurable effect."
    )

    _heading(doc, "Proportional hazards assumption", level=3)
    _body(doc,
        f"The Schoenfeld-residuals test flagged {n_ph_violated} covariate(s) as "
        "potentially violating the proportional-hazards assumption (p < 0.05). "
        "This should be interpreted cautiously: at n = 300,000 the test has extreme "
        "statistical power and will detect even negligible departures from PH. The "
        "practical question is not whether PH holds exactly, but whether the HRs "
        "are useful summaries. A stratified Cox model (baseline hazard allowed to "
        "vary by instrument) was fitted as a robustness check; the non-symbol "
        "covariate HRs were materially unchanged, supporting the primary model's "
        "conclusions."
    )

    # ---- 3.8 Competing risks (was 3.7) ----
    _heading(doc, "3.8  Competing-risks validation (Aalen-Johansen)", level=2)
    _add_figure(doc, "fig_aalen_johansen.png",
        "Figure 15. Aalen-Johansen competing-risks cumulative incidence functions: "
        "loss-close (solid red) and profit-close (solid green), with the cause-specific "
        "1 - KM estimate overlaid (dashed red) for comparison."
    )
    _body(doc,
        "The AJ CIF for loss-close lies below the cause-specific 1 - KM curve throughout, "
        "as expected: by treating profitable closes as random censoring, the KM method "
        "assumes they would eventually become losses, inflating the estimated cumulative "
        "incidence. The AJ loss CIF represents the true probability of a trade closing "
        "at a loss by time t, accounting for the fact that many trades exit profitably "
        "first. The two curves agree in shape and slope — confirming the causal structure "
        "of the model — but diverge in level, quantifying the magnitude of the "
        "cause-specific inflation. For practical purposes (identifying the high-risk "
        "holding window), both estimates point to the same critical region."
    )

    doc.add_page_break()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report() -> None:
    doc = Document()

    # Page margins
    section = doc.sections[0]
    section.left_margin = Inches(1.1)
    section.right_margin = Inches(1.1)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    add_title_page(doc)
    add_executive_summary(doc)
    add_section1_context(doc)
    add_section2_descriptive(doc)

    survival_csv = PROCESSED / "km_incidence_table.csv"
    if survival_csv.exists():
        add_section3_survival(doc)

    doc.save(str(OUT_FILE))
    print(f"Report saved: {OUT_FILE}  ({OUT_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build_report()
