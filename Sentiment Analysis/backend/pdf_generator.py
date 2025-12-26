from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

def generate_pdf(company_name, results):
    file_path = f"{company_name}_sentiment_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph(f"<b>BizChain AI Sentiment Report</b>", styles["Title"]))
    story.append(Paragraph(f"Company/Product: {company_name}", styles["Heading2"]))
    story.append(Spacer(1,12))

    for item in results:
        story.append(Paragraph(f"<b>Comment:</b> {item['comment']}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Sentiment:</b> {item['sentiment']}", styles["BodyText"]))
        story.append(Spacer(1, 12))

    doc.build(story)

    return file_path
