from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


# =============================================================================
# Brand settings (tweak these to match BioERGOtech)
# =============================================================================
BRAND = {
    # Choose 1-2 brand tones. These are safe defaults if you haven't extracted exact hex.
    "primary": "#0F3D3E",   # deep teal-ish
    "accent":  "#2D6A4F",   # green-ish
    "light":   "#F2F5F7",
    "text":    "#1C1C1C",
    "muted":   "#6B7280",
}


# =============================================================================
# Formatting helpers
# =============================================================================
def _fmt_prob(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return "NA"
    if pd.isna(p):
        return "NA"
    if p == 0.0:
        return "≈ 0 (very small)"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.6f}"


def _risk_color(stratum: str):
    # Keep these clinical colors recognizable
    if stratum == "Low":
        return colors.HexColor("#2e7d32")
    if stratum == "Medium":
        return colors.HexColor("#ed6c02")
    if stratum == "High":
        return colors.HexColor("#d32f2f")
    return colors.HexColor("#444444")


def _safe_img(png_bytes: bytes, width_cm: float, height_cm: float) -> Optional[Image]:
    if not png_bytes:
        return None
    img_buf = BytesIO(png_bytes)
    img = Image(img_buf)
    img.drawWidth = width_cm * cm
    img.drawHeight = height_cm * cm
    return img


def _styled_table(
    data: List[List[Any]],
    col_widths_cm: List[float],
    *,
    header_bg: str = "#F2F2F2",
    font_size: int = 9,
    repeat_rows: int = 1,
) -> Table:
    t = Table(data, colWidths=[w * cm for w in col_widths_cm], hAlign="LEFT", repeatRows=repeat_rows)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(BRAND["text"])),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(BRAND["text"])),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _header_bar(title: str, subtitle: str = "") -> Table:
    """
    Create a simple brand header bar as a Table so it renders consistently.
    """
    data = [[title], [subtitle]] if subtitle else [[title]]
    t = Table(data, colWidths=[17.0 * cm], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BRAND["primary"])),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 14),
                ("FONTSIZE", (0, 1), (-1, 1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


# =============================================================================
# Data structure (MATCHES YOUR RESULTS PAGE)
# =============================================================================
@dataclass
class PDFInputs:
    patient_id: Optional[str]
    notes: Optional[str]

    age_group: Optional[str]
    gender: Optional[str]
    ethnicity: Optional[str]
    education_level: Optional[str]
    employment_status: Optional[str]

    # scoring outputs
    vero_probability: float
    vero_probability_display: float
    vero_score: int
    risk_stratum: str

    # completeness
    fields_provided: int
    fields_total: int

    # phenotype label + threshold used
    phenotype_label: str
    phenotype_threshold: float


# =============================================================================
# PDF builder
# =============================================================================
def build_vero_pdf(
    summary: PDFInputs,
    top_contributors: pd.DataFrame,
    contrib_bar_png_bytes: bytes,
    gauge_png_bytes: bytes,
    timeline_png_bytes: bytes,
    timeline_events: List[Dict[str, Any]],
    *,
    assets_dir: Optional[Path] = None,
) -> bytes:
    """
    Returns PDF bytes.

    assets_dir (optional): if provided, attempts to embed a logo.
    Expected file name examples:
    - bioergotech_logo.png
    - logo.png
    """
    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.6 * cm,
        title="VERO Patient Summary",
        author="BioERGOtech - VERO Risk Calculator",
    )

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].fontSize = 10
    styles["Normal"].textColor = colors.HexColor(BRAND["text"])

    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor(BRAND["primary"]),
        spaceBefore=6,
        spaceAfter=6,
    )

    small_grey = ParagraphStyle(
        "small_grey",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor(BRAND["muted"]),
    )

    story: List[Any] = []

    # -------------------------------------------------------------------------
    # Header: brand bar + optional logo
    # -------------------------------------------------------------------------
    story.append(_header_bar("VERO Patient Summary", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"))

    # optional logo
    if assets_dir is not None:
        candidates = [
            assets_dir / "bioergotech_logo.png",
            assets_dir / "logo.png",
            assets_dir / "bioergotech_logo.jpg",
            assets_dir / "logo.jpg",
        ]
        logo_path = next((p for p in candidates if p.exists()), None)
        if logo_path:
            try:
                logo = Image(str(logo_path))
                logo.drawWidth = 5.0 * cm
                logo.drawHeight = 1.5 * cm
                story.append(Spacer(1, 0.25 * cm))
                story.append(logo)
            except Exception:
                pass

    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # Patient details
    # -------------------------------------------------------------------------
    story.append(Paragraph("Patient details", h2))

    pid = summary.patient_id or "Not provided"
    demo_rows = [
        ["Field", "Value"],
        ["Patient ID", pid],
        ["Age group", summary.age_group or "Unknown"],
        ["Gender", summary.gender or "Unknown"],
        ["Ethnicity", summary.ethnicity or "Unknown"],
        ["Education level", summary.education_level or "Unknown"],
        ["Employment status", summary.employment_status or "Unknown"],
    ]
    story.append(_styled_table(demo_rows, col_widths_cm=[5.2, 11.0], header_bg=BRAND["light"], font_size=9))
    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # Risk summary
    # -------------------------------------------------------------------------
    story.append(Paragraph("Risk summary (VERO score)", h2))
    risk_col = _risk_color(summary.risk_stratum)

    risk_rows = [
        ["Metric", "Value"],
        ["Probability used for score", _fmt_prob(summary.vero_probability)],
        ["VERO score (0-100)", str(int(summary.vero_score))],
        ["Risk stratum", str(summary.risk_stratum)],
        ["Fields provided", f"{int(summary.fields_provided)}/{int(summary.fields_total)}"],
    ]

    score_table = Table(risk_rows, colWidths=[7.2 * cm, 9.0 * cm], hAlign="LEFT")
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND["light"])),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                # highlight risk row
                ("BACKGROUND", (0, 3), (-1, 3), risk_col),
                ("TEXTCOLOR", (0, 3), (-1, 3), colors.white),
                ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"),
            ]
        )
    )
    story.append(score_table)
    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # Phenotype membership
    # -------------------------------------------------------------------------
    story.append(Paragraph("Phenotype membership", h2))
    story.append(Paragraph(f"Predicted membership: <b>{summary.phenotype_label}</b>", styles["Normal"]))
    story.append(Paragraph(f"Threshold used: {float(summary.phenotype_threshold):.2f}", styles["Normal"]))
    story.append(Spacer(1, 0.2 * cm))

    gauge_img = _safe_img(gauge_png_bytes, width_cm=16.0, height_cm=5.6)
    if gauge_img is not None:
        story.append(gauge_img)
        story.append(Spacer(1, 0.30 * cm))
    else:
        story.append(Paragraph("Gauge not available.", small_grey))
        story.append(Spacer(1, 0.15 * cm))

    # -------------------------------------------------------------------------
    # Contributors
    # -------------------------------------------------------------------------
    story.append(Paragraph("Top contributors", h2))

    tdf = top_contributors.copy()
    required_cols = ["feature_code", "feature_label", "total_contribution"]
    for c in required_cols:
        if c not in tdf.columns:
            raise ValueError(f"Missing column in top_contributors: {c}")

    tdf["total_contribution"] = pd.to_numeric(tdf["total_contribution"], errors="coerce").fillna(0.0).round(4)

    table_data = [required_cols] + tdf[required_cols].values.tolist()
    story.append(_styled_table(table_data, col_widths_cm=[5.0, 7.0, 3.4], header_bg=BRAND["light"], font_size=9))
    story.append(Spacer(1, 0.30 * cm))

    story.append(Paragraph("Contributor chart", h2))
    contrib_img = _safe_img(contrib_bar_png_bytes, width_cm=16.0, height_cm=6.8)
    if contrib_img is not None:
        story.append(contrib_img)
        story.append(Spacer(1, 0.30 * cm))
    else:
        story.append(Paragraph("Contributor chart not available.", small_grey))
        story.append(Spacer(1, 0.15 * cm))

    # -------------------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------------------
    story.append(Paragraph("Patient timeline", h2))

    timeline_img = _safe_img(timeline_png_bytes, width_cm=16.0, height_cm=5.8)
    if timeline_img is not None:
        story.append(timeline_img)
        story.append(Spacer(1, 0.20 * cm))
    else:
        story.append(Paragraph("Timeline chart not available.", small_grey))
        story.append(Spacer(1, 0.15 * cm))

    if timeline_events:
        ev_df = pd.DataFrame(timeline_events).copy()
        if "date" in ev_df.columns:
            ev_df["date_str"] = ev_df["date"].astype(str)
        else:
            ev_df["date_str"] = ""
        if "event" not in ev_df.columns:
            ev_df["event"] = ""

        ev_df = ev_df[["event", "date_str"]].copy()
        tdata = [["Event", "Date"]] + ev_df.values.tolist()
        story.append(_styled_table(tdata, col_widths_cm=[9.0, 6.6], header_bg=BRAND["light"], font_size=9))
    else:
        story.append(Paragraph("No timeline events provided.", styles["Normal"]))

    story.append(Spacer(1, 0.35 * cm))

    # -------------------------------------------------------------------------
    # Notes + disclaimer
    # -------------------------------------------------------------------------
    story.append(Paragraph("Clinical notes", h2))
    story.append(Paragraph(summary.notes or "None", styles["Normal"]))
    story.append(Spacer(1, 0.55 * cm))

    story.append(
        Paragraph(
            "Decision support only. Interpret alongside clinical judgment.",
            small_grey,
        )
    )

    doc.build(story)
    return buf.getvalue()
