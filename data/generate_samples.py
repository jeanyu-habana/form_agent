"""Generate synthetic sample forms for demos and tests.

Produces ~5 PDF forms in data/samples/:
- 2 job applications
- 1 medical intake
- 2 expense reports
One of the expense reports is rendered as an image-PDF to exercise the OCR fallback.
"""
from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

OUT_DIR = Path(__file__).parent / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_pdf(path: Path, title: str, rows: list[tuple[str, str]], notes: str = "") -> None:
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title=title)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"<b>{title}</b>", styles["Title"]), Spacer(1, 12)]
    table = Table([[k, v] for k, v in rows], colWidths=[180, 320])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    if notes:
        story += [Spacer(1, 16), Paragraph("<b>Notes</b>", styles["Heading3"]),
                  Paragraph(notes.replace("\n", "<br/>"), styles["BodyText"])]
    doc.build(story)


def _build_image_pdf(path: Path, title: str, rows: list[tuple[str, str]], notes: str = "") -> None:
    """Render the form as text on an image, then embed the image into a PDF.

    This produces a 'scanned-style' PDF whose text is NOT extractable by pypdf,
    forcing the OCR fallback path.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Pillow required to build scanned-style sample") from e

    width, height = 1700, 2200  # ~ letter at 200dpi
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("Helvetica.ttc", 48)
        font = ImageFont.truetype("Helvetica.ttc", 28)
    except OSError:
        font_title = ImageFont.load_default()
        font = ImageFont.load_default()

    y = 80
    draw.text((80, y), title, fill="black", font=font_title)
    y += 100
    for k, v in rows:
        draw.text((80, y), f"{k}:", fill="black", font=font)
        draw.text((620, y), v, fill="black", font=font)
        y += 50
    if notes:
        y += 40
        draw.text((80, y), "Notes:", fill="black", font=font)
        y += 50
        for line in notes.split("\n"):
            draw.text((80, y), line, fill="black", font=font)
            y += 40

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    c = canvas.Canvas(str(path), pagesize=LETTER)
    pw, ph = LETTER
    c.drawImage(ImageReader(buf), 0, 0, width=pw, height=ph)
    c.showPage()
    c.save()


SAMPLES = [
    ("job_app_01.pdf", "Job Application", False, [
        ("Applicant Name", "Maya Chen"),
        ("Email", "maya.chen@example.com"),
        ("Phone", "+1-415-555-0142"),
        ("Position Applied For", "Senior Backend Engineer"),
        ("Years of Experience", "9"),
        ("Most Recent Employer", "Acme Cloud Systems"),
        ("Most Recent Title", "Staff Engineer"),
        ("Education", "M.S. Computer Science, UC Berkeley, 2017"),
        ("Available Start Date", "2026-06-01"),
        ("Salary Expectation (USD)", "215000"),
        ("Authorized to Work in US", "Yes"),
    ], "Strong background in distributed systems and Kubernetes. Open to hybrid work in SF Bay Area."),

    ("job_app_02.pdf", "Job Application", False, [
        ("Applicant Name", "Devon Park"),
        ("Email", "devon.park@example.com"),
        ("Phone", "+1-206-555-0177"),
        ("Position Applied For", "Senior Backend Engineer"),
        ("Years of Experience", "4"),
        ("Most Recent Employer", "Northwind Logistics"),
        ("Most Recent Title", "Software Engineer II"),
        ("Education", "B.S. Computer Engineering, U. Washington, 2021"),
        ("Available Start Date", "2026-05-15"),
        ("Salary Expectation (USD)", "175000"),
        ("Authorized to Work in US", "Yes"),
    ], "Looking for high-impact infra role. Comfortable with Go and Python."),

    ("medical_intake_01.pdf", "Patient Intake Form", False, [
        ("Patient Name", "Aaliyah Johnson"),
        ("Date of Birth", "1988-03-22"),
        ("Sex", "Female"),
        ("Visit Date", "2026-04-19"),
        ("Primary Complaint", "Persistent migraine for 3 weeks"),
        ("Allergies", "Penicillin"),
        ("Current Medications", "Sumatriptan 50mg PRN; Lisinopril 10mg daily"),
        ("Past Medical History", "Hypertension (2019), Appendectomy (2010)"),
        ("Family History", "Father: type 2 diabetes; Mother: hypertension"),
        ("Smoker", "No"),
        ("Alcohol Use (drinks/week)", "2"),
        ("Emergency Contact", "Marcus Johnson +1-510-555-0188"),
    ], "Patient reports photosensitivity and nausea accompanying headaches. No recent head trauma."),

    ("expense_report_01.pdf", "Expense Report", False, [
        ("Employee Name", "Maya Chen"),
        ("Employee ID", "E-10472"),
        ("Department", "Engineering"),
        ("Report Period", "2026-03-01 to 2026-03-31"),
        ("Project Code", "PRJ-INFRA-22"),
        ("Total Amount (USD)", "1842.55"),
        ("Currency", "USD"),
        ("Reimbursement Method", "Direct Deposit"),
        ("Submitted Date", "2026-04-02"),
        ("Approved By", "Pending"),
    ], "Items: airfare SFO->SEA $612.40; hotel 3 nights $789.00; meals $241.15; rideshare $200.00."),

    ("expense_report_02_scanned.pdf", "Expense Report", True, [
        ("Employee Name", "Devon Park"),
        ("Employee ID", "E-22310"),
        ("Department", "Platform"),
        ("Report Period", "2026-03-15 to 2026-03-29"),
        ("Project Code", "PRJ-PLATFORM-08"),
        ("Total Amount (USD)", "534.20"),
        ("Currency", "USD"),
        ("Reimbursement Method", "Payroll"),
        ("Submitted Date", "2026-04-01"),
        ("Approved By", "L. Park"),
    ], "Items: client lunch $112.30; conference fee $349.00; taxi $72.90."),
]


def main() -> None:
    for filename, title, scanned, rows, *rest in SAMPLES:
        notes = rest[0] if rest else ""
        out = OUT_DIR / filename
        if scanned:
            _build_image_pdf(out, title, rows, notes)
        else:
            _build_pdf(out, title, rows, notes)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
