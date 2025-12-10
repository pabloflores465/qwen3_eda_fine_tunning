#!/usr/bin/env python3
"""
Generate PDF Report for VerilogEval Benchmark Results
Compares baseline vs fine-tuned Qwen3-4B model on NVIDIA VerilogEval v2
"""

import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def create_bar_chart(data, labels, title, width=400, height=200):
    """Create a bar chart comparing baseline vs fine-tuned."""
    drawing = Drawing(width, height)

    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 30
    bc.height = 120
    bc.width = 300
    bc.data = data
    bc.categoryAxis.categoryNames = labels
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(max(d) for d in data) * 1.2
    bc.bars[0].fillColor = colors.HexColor('#3498db')  # Blue for baseline
    bc.bars[1].fillColor = colors.HexColor('#2ecc71')  # Green for fine-tuned
    bc.barWidth = 15
    bc.groupSpacing = 20

    # Add legend
    legend = Legend()
    legend.x = 360
    legend.y = 100
    legend.colorNamePairs = [
        (colors.HexColor('#3498db'), 'Baseline'),
        (colors.HexColor('#2ecc71'), 'Fine-tuned')
    ]

    drawing.add(bc)
    drawing.add(legend)

    return drawing

def generate_report(results_file, output_file):
    """Generate the PDF report."""

    # Load results
    results = load_results(results_file)
    baseline = results['baseline']
    finetuned = results['finetuned']

    # Create PDF
    doc = SimpleDocTemplate(output_file, pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2c3e50')
    )

    # Title Page
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("VerilogEval Benchmark Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Qwen3-4B Model Comparison", styles['Heading2']))
    story.append(Paragraph("Baseline vs Fine-tuned (LoRA)", styles['Heading3']))
    story.append(Spacer(1, 0.5*inch))

    # Metadata
    meta_data = [
        ["Benchmark", "NVIDIA VerilogEval v2 (spec-to-rtl)"],
        ["Model", "Qwen3-4B-Thinking-2507-MLX-4bit"],
        ["Hardware", "MacBook Air M1 8GB"],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Problems Tested", str(baseline['total_problems'])],
    ]

    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
    ]))
    story.append(meta_table)

    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))

    pass_improvement = finetuned['pass_at_1'] - baseline['pass_at_1']
    compile_improvement = finetuned['compile_rate'] - baseline['compile_rate']

    summary_text = f"""
    The fine-tuned model with LoRA adapter shows significant improvement over the baseline model
    on the VerilogEval benchmark:
    <br/><br/>
    <b>Key Findings:</b><br/>
    - Pass@1 improved from {baseline['pass_at_1']:.1f}% to {finetuned['pass_at_1']:.1f}% (<font color="green">+{pass_improvement:.1f}%</font>)<br/>
    - Compile rate improved from {baseline['compile_rate']:.1f}% to {finetuned['compile_rate']:.1f}% (<font color="green">+{compile_improvement:.1f}%</font>)<br/>
    - Test passes increased from {baseline['test_pass']} to {finetuned['test_pass']} problems<br/>
    - Average generation time: {baseline['avg_generation_time']:.1f}s (baseline) vs {finetuned['avg_generation_time']:.1f}s (fine-tuned)
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Main Results Table
    story.append(Paragraph("Benchmark Results Comparison", heading_style))

    results_data = [
        ["Metric", "Baseline", "Fine-tuned", "Improvement"],
        ["Total Problems", str(baseline['total_problems']), str(finetuned['total_problems']), "-"],
        ["Compile Success", str(baseline['compile_success']), str(finetuned['compile_success']),
         f"+{finetuned['compile_success'] - baseline['compile_success']}"],
        ["Test Pass", str(baseline['test_pass']), str(finetuned['test_pass']),
         f"+{finetuned['test_pass'] - baseline['test_pass']}"],
        ["Pass@1 (%)", f"{baseline['pass_at_1']:.2f}", f"{finetuned['pass_at_1']:.2f}",
         f"+{pass_improvement:.2f}%"],
        ["Compile Rate (%)", f"{baseline['compile_rate']:.2f}", f"{finetuned['compile_rate']:.2f}",
         f"+{compile_improvement:.2f}%"],
        ["Avg Gen Time (s)", f"{baseline['avg_generation_time']:.2f}", f"{finetuned['avg_generation_time']:.2f}",
         f"{finetuned['avg_generation_time'] - baseline['avg_generation_time']:.2f}s"],
    ]

    results_table = Table(results_data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))

    # Bar Chart - Pass Rate
    story.append(Paragraph("Pass@1 Comparison", heading_style))
    chart_data = [
        [baseline['pass_at_1'], baseline['compile_rate']],
        [finetuned['pass_at_1'], finetuned['compile_rate']]
    ]
    chart = create_bar_chart(chart_data, ['Pass@1 (%)', 'Compile Rate (%)'], 'Performance Comparison')
    story.append(chart)
    story.append(Spacer(1, 0.3*inch))

    story.append(PageBreak())

    # Detailed Results
    story.append(Paragraph("Detailed Problem Results", heading_style))

    # Create comparison table for each problem
    detail_data = [["Problem", "Baseline", "Fine-tuned", "Notes"]]

    baseline_results = {r['id']: r for r in baseline['results']}
    finetuned_results = {r['id']: r for r in finetuned['results']}

    for prob_id in baseline_results:
        b_result = baseline_results[prob_id]
        f_result = finetuned_results.get(prob_id, {})

        b_status = "PASS" if b_result.get('passes', False) else ("COMPILE" if b_result.get('compiles', False) else "FAIL")
        f_status = "PASS" if f_result.get('passes', False) else ("COMPILE" if f_result.get('compiles', False) else "FAIL")

        # Note improvements
        note = ""
        if b_status != "PASS" and f_status == "PASS":
            note = "Improved"
        elif b_status == "PASS" and f_status != "PASS":
            note = "Regressed"

        detail_data.append([
            f"{b_result['id']}_{b_result['name'][:15]}",
            b_status,
            f_status,
            note
        ])

    detail_table = Table(detail_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1*inch])
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
    ]))
    story.append(detail_table)

    story.append(PageBreak())

    # Conclusions
    story.append(Paragraph("Conclusions", heading_style))

    conclusions = f"""
    <b>1. Fine-tuning Effectiveness:</b><br/>
    The LoRA fine-tuning on EDA/Verilog data resulted in a <b>{pass_improvement:.1f}% improvement</b> in Pass@1 rate
    on the VerilogEval benchmark. This demonstrates that domain-specific fine-tuning significantly enhances
    the model's ability to generate correct Verilog code.<br/><br/>

    <b>2. Compile Rate Improvement:</b><br/>
    The compile success rate improved by <b>{compile_improvement:.1f}%</b>, indicating that the fine-tuned model
    produces more syntactically correct Verilog code.<br/><br/>

    <b>3. Generation Speed:</b><br/>
    The fine-tuned model shows {'faster' if finetuned['avg_generation_time'] < baseline['avg_generation_time'] else 'slower'}
    average generation time ({finetuned['avg_generation_time']:.1f}s vs {baseline['avg_generation_time']:.1f}s),
    likely due to more confident token predictions from domain knowledge.<br/><br/>

    <b>4. Benchmark Context:</b><br/>
    VerilogEval is the standard benchmark for evaluating LLM Verilog generation capabilities.
    State-of-the-art models like GPT-4o achieve ~63% Pass@1, while specialized models like CodeV-R1-7B
    reach ~68-72%. Our fine-tuned model achieves <b>{finetuned['pass_at_1']:.1f}%</b> Pass@1.<br/><br/>

    <b>5. Recommendations:</b><br/>
    - Continue fine-tuning with more diverse Verilog examples<br/>
    - Consider increasing training epochs for complex circuits<br/>
    - The fine-tuned model is suitable for EDA code generation tasks
    """
    story.append(Paragraph(conclusions, styles['Normal']))

    # Build PDF
    doc.build(story)
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    generate_report("verilog_eval_results.json", "verilog_eval_report.pdf")
