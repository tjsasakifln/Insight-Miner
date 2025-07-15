from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io

def create_sentiment_chart(positive: float, negative: float, neutral: float):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive, negative, neutral]
    colors = ['#4CAF50', '#F44336', '#FFEB3B']
    explode = (0.1, 0, 0)  # explode 1st slice

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig1)
    return buf

def generate_pdf_report(data: dict, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Insight Miner - Relatório de Análise", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(f"Data do Relatório: {data.get("report_date", "N/A")}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Resumo da Análise de Sentimento", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Total de Reviews Processados: {data.get("total_reviews", 0)}", styles['Normal']))
    story.append(Paragraph(f"Reviews Positivos: {data.get("positive_reviews", 0)}", styles['Normal']))
    story.append(Paragraph(f"Reviews Negativos: {data.get("negative_reviews", 0)}", styles['Normal']))
    story.append(Paragraph(f"Reviews Neutros: {data.get("neutral_reviews", 0)}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Add sentiment chart
    positive_perc = data.get("positive_reviews", 0) / data.get("total_reviews", 1) * 100
    negative_perc = data.get("negative_reviews", 0) / data.get("total_reviews", 1) * 100
    neutral_perc = data.get("neutral_reviews", 0) / data.get("total_reviews", 1) * 100
    chart_buffer = create_sentiment_chart(positive_perc, negative_perc, neutral_perc)
    story.append(Image(chart_buffer, width=4*inch, height=3*inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Métricas de Negócio", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Total de Insights Gerados: {data.get("total_insights", 0)}", styles['Normal']))
    story.append(Paragraph(f"Tempo Médio de Processamento: {data.get("avg_processing_time", "N/A")}", styles['Normal']))
    story.append(Paragraph(f"ROI Estimado: {data.get("estimated_roi", "N/A")}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Tópicos em Tendência", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    for topic in data.get("trending_topics", []):
        story.append(Paragraph(f"- {topic}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
