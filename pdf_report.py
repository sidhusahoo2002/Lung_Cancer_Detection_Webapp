from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

def generate_pdf(name, prediction, benign, malignant):
    os.makedirs("reports", exist_ok=True)
    path = f"reports/{name}_report.pdf"

    c = canvas.Canvas(path, pagesize=A4)
    c.drawString(50, 800, "LUNG CANCER DIAGNOSIS REPORT")
    c.drawString(50, 760, f"Patient Name: {name}")
    c.drawString(50, 720, f"Prediction: {prediction}")
    c.drawString(50, 700, f"Benign Probability: {benign:.2f}")
    c.drawString(50, 680, f"Malignant Probability: {malignant:.2f}")
    c.drawString(50, 640, "Disclaimer: For educational use only")

    c.save()
    return path