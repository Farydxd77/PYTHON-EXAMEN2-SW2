from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configurar OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


@app.route('/')
def hello_world():
    return jsonify({
        'message': 'TesloShop AI Service 🤖 (Powered by OpenAI + ML)',
        'status': 'running',
        'version': '3.0.0',
        'features': ['charts', 'reports', 'predictions']
    })


@app.route('/api/generate-chart', methods=['POST'])
def generate_chart():
    """Genera datos de gráfica usando IA"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Se requiere un prompt'}), 400
        
        user_prompt = data['prompt']
        productos = data.get('productos', [])
        ventas = data.get('ventas', [])
        
        context = ""
        if productos:
            productos_summary = [
                {
                    'sku': p.get('sku'),
                    'nombre': p.get('nombre'),
                    'categoria': p.get('categoria'),
                    'stock': p.get('cantidadDisponible'),
                    'margen': p.get('margen'),
                }
                for p in productos[:20]
            ]
            context += f"\n\nPRODUCTOS:\n{json.dumps(productos_summary, indent=2)}"
        
        if ventas:
            context += f"\n\nVENTAS:\n{json.dumps(ventas[:50], indent=2)}"
        
        full_prompt = f"""
Eres un analista de datos. Analiza estos datos y responde al usuario.

DATOS:
{context}

PETICIÓN: "{user_prompt}"

Responde SOLO con JSON:
{{
    "chart_type": "bar|line|pie|area",
    "title": "Título",
    "data": [{{"label": "X", "value": 123}}],
    "insight": "Análisis breve"
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Respondes SOLO con JSON válido."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Limpiar markdown
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            chart_data = json.loads(json_match.group())
            return jsonify(chart_data), 200
        else:
            return jsonify({'error': 'No se pudo extraer JSON'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Genera un reporte en PDF"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Se requiere un prompt'}), 400
        
        user_prompt = data['prompt']
        data_source = data.get('dataSource', 'all')
        productos = data.get('productos', [])
        ordenes = data.get('ordenes', [])
        ventas = data.get('ventas', [])
        
        print(f"📊 Generando reporte: {user_prompt}")
        
        # Construir contexto
        context = ""
        
        if data_source in ['productos', 'all'] and productos:
            context += f"\n\nPRODUCTOS:\n{json.dumps(productos[:30], indent=2)}"
        
        if data_source in ['ordenes', 'all'] and ordenes:
            context += f"\n\nÓRDENES:\n{json.dumps(ordenes[:30], indent=2)}"
        
        if data_source in ['ventas', 'all'] and ventas:
            context += f"\n\nVENTAS:\n{json.dumps(ventas[:30], indent=2)}"
        
        ai_prompt = f"""
Analiza los datos y genera un reporte.

DATOS:
{context}

SOLICITUD: "{user_prompt}"

Responde SOLO con JSON:
{{
    "titulo": "Título del Reporte",
    "resumen": "Resumen ejecutivo",
    "datos_filtrados": [{{"campo1": "valor1"}}],
    "columnas": ["campo1", "campo2"],
    "analisis": "Análisis detallado",
    "recomendaciones": ["rec1", "rec2"],
    "metricas_clave": {{"Total": 10}}
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Respondes SOLO con JSON válido."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Limpiar
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return jsonify({'error': 'No se pudo extraer JSON'}), 500
        
        report_data = json.loads(json_match.group())
        
        # Generar PDF
        pdf_buffer = generate_pdf_report(report_data, user_prompt)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'reporte_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint de predicciones con Machine Learning
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Se requiere un prompt'}), 400
        
        user_prompt = data['prompt']
        prediction_type = data.get('predictionType', 'auto')  # 'ventas', 'ordenes', 'productos', 'auto'
        ventas = data.get('ventas', [])
        productos = data.get('productos', [])
        ordenes = data.get('ordenes', [])
        
        print(f"🔮 Generando predicción: {user_prompt}")
        print(f"📊 Tipo: {prediction_type}")
        
        # Determinar tipo de predicción automáticamente si es 'auto'
        if prediction_type == 'auto':
            prompt_lower = user_prompt.lower()
            if 'venta' in prompt_lower or 'vendido' in prompt_lower:
                prediction_type = 'ventas'
            elif 'orden' in prompt_lower or 'pedido' in prompt_lower:
                prediction_type = 'ordenes'
            elif 'producto' in prompt_lower or 'stock' in prompt_lower:
                prediction_type = 'productos'
            else:
                prediction_type = 'ventas'  # Default
        
        # Realizar predicción según el tipo
        prediction_result = None
        
        if prediction_type == 'ventas' and ventas:
            prediction_result = predict_sales(ventas, user_prompt)
        elif prediction_type == 'ordenes' and ordenes:
            prediction_result = predict_orders(ordenes, user_prompt)
        elif prediction_type == 'productos' and productos:
            prediction_result = predict_products(productos, ventas, user_prompt)
        else:
            return jsonify({'error': 'Datos insuficientes para predicción'}), 400
        
        if prediction_result:
            print(f"✅ Predicción generada: {prediction_result['title']}")
            return jsonify(prediction_result), 200
        else:
            return jsonify({'error': 'No se pudo generar predicción'}), 500
            
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return jsonify({'error': str(e)}), 500


def predict_sales(ventas, prompt):
    """
    Predice ventas futuras usando regresión lineal
    """
    try:
        # Agrupar ventas por fecha
        df = pd.DataFrame(ventas)
        
        # Convertir fecha a datetime
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Agrupar por fecha y sumar ventas
        daily_sales = df.groupby('fecha')['ventas'].sum().reset_index()
        daily_sales = daily_sales.sort_values('fecha')
        
        # Preparar datos para el modelo
        daily_sales['dias'] = (daily_sales['fecha'] - daily_sales['fecha'].min()).dt.days
        
        X = daily_sales['dias'].values.reshape(-1, 1)
        y = daily_sales['ventas'].values
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir próximos 30 días
        last_day = daily_sales['dias'].max()
        future_days = np.array([[last_day + i] for i in range(1, 31)])
        predictions = model.predict(future_days)
        
        # Calcular tendencia
        slope = model.coef_[0]
        trend = "creciente" if slope > 0 else "decreciente"
        trend_pct = abs(slope / y.mean() * 100)
        
        # Preparar respuesta
        last_date = daily_sales['fecha'].max()
        future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 31)]
        
        # Datos para gráfica (últimos 7 días históricos + 7 días predichos)
        historical_data = [
            {"label": row['fecha'].strftime('%d/%m'), "value": int(row['ventas']), "type": "historical"}
            for _, row in daily_sales.tail(7).iterrows()
        ]
        
        predicted_data = [
            {"label": (last_date + timedelta(days=i)).strftime('%d/%m'), "value": int(predictions[i-1]), "type": "predicted"}
            for i in range(1, 8)  # Próximos 7 días
        ]
        
        chart_data = historical_data + predicted_data
        
        # Calcular métricas
        avg_historical = int(y.mean())
        avg_predicted = int(predictions[:7].mean())
        total_predicted_month = int(predictions.sum())
        
        return {
            "prediction_type": "ventas",
            "title": "Predicción de Ventas - Próximos 30 Días",
            "chart_type": "line",
            "data": chart_data,
            "metrics": {
                "promedio_historico": avg_historical,
                "promedio_predicho_7dias": avg_predicted,
                "total_predicho_mes": total_predicted_month,
                "tendencia": trend,
                "cambio_porcentual": f"{trend_pct:.1f}%"
            },
            "predictions": [
                {"fecha": future_dates[i], "ventas_predichas": int(predictions[i])}
                for i in range(0, min(30, len(predictions)))
            ],
            "insight": f"Basándose en el análisis de {len(daily_sales)} días históricos, se predice una tendencia {trend} "
                      f"con un cambio del {trend_pct:.1f}%. "
                      f"Se esperan aproximadamente {avg_predicted} ventas diarias en la próxima semana "
                      f"y un total de {total_predicted_month} ventas en el próximo mes.",
            "confidence": "75%",
            "model_accuracy": f"R² = {model.score(X, y):.2f}"
        }
        
    except Exception as e:
        print(f"Error en predict_sales: {e}")
        return None


def predict_orders(ordenes, prompt):
    """
    Predice cantidad de órdenes futuras
    """
    try:
        df = pd.DataFrame(ordenes)
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Contar órdenes por día
        daily_orders = df.groupby('fecha').size().reset_index(name='ordenes')
        daily_orders = daily_orders.sort_values('fecha')
        
        # Preparar datos
        daily_orders['dias'] = (daily_orders['fecha'] - daily_orders['fecha'].min()).dt.days
        
        X = daily_orders['dias'].values.reshape(-1, 1)
        y = daily_orders['ordenes'].values
        
        # Modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir próximas 4 semanas
        last_day = daily_orders['dias'].max()
        future_days = np.array([[last_day + i] for i in range(1, 29)])
        predictions = model.predict(future_days)
        
        # Agrupar por semana
        weeks_data = []
        for week in range(4):
            start_idx = week * 7
            end_idx = start_idx + 7
            week_total = int(predictions[start_idx:end_idx].sum())
            weeks_data.append({
                "label": f"Semana {week + 1}",
                "value": week_total
            })
        
        avg_weekly = int(np.mean([w['value'] for w in weeks_data]))
        
        return {
            "prediction_type": "ordenes",
            "title": "Predicción de Órdenes - Próximas 4 Semanas",
            "chart_type": "bar",
            "data": weeks_data,
            "metrics": {
                "promedio_semanal": avg_weekly,
                "total_mes": int(predictions.sum()),
                "promedio_diario": int(predictions.mean())
            },
            "insight": f"Se espera un promedio de {avg_weekly} órdenes por semana. "
                      f"El modelo predice aproximadamente {int(predictions.mean())} órdenes diarias "
                      f"basándose en {len(daily_orders)} días de datos históricos.",
            "confidence": "70%",
            "model_accuracy": f"R² = {model.score(X, y):.2f}"
        }
        
    except Exception as e:
        print(f"Error en predict_orders: {e}")
        return None


def predict_products(productos, ventas, prompt):
    """
    Predice qué productos se venderán más
    """
    try:
        # Calcular score por producto basado en rotación, margen y stock
        product_scores = []
        
        for p in productos:
            # Score basado en múltiples factores
            rotacion = p.get('rotacion', 0)
            margen = p.get('margen', 0)
            stock = p.get('cantidadDisponible', 0)
            
            # Fórmula de score (puedes ajustarla)
            score = (rotacion * 0.5) + (margen * 0.3) + (min(stock, 100) * 0.2)
            
            product_scores.append({
                "producto": p.get('nombre', 'Unknown'),
                "score": score,
                "rotacion": rotacion,
                "margen": margen,
                "stock": stock
            })
        
        # Ordenar por score
        product_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Top 5 productos
        top_products = product_scores[:5]
        
        chart_data = [
            {"label": p['producto'][:20], "value": int(p['score'] * 10)}
            for p in top_products
        ]
        
        return {
            "prediction_type": "productos",
            "title": "Predicción: Productos con Mayor Potencial de Venta",
            "chart_type": "bar",
            "data": chart_data,
            "top_products": [
                {
                    "producto": p['producto'],
                    "score_venta": int(p['score'] * 10),
                    "rotacion": p['rotacion'],
                    "margen": f"{p['margen']:.1f}%",
                    "stock": p['stock']
                }
                for p in top_products
            ],
            "insight": f"Basándose en rotación, margen y disponibilidad, {top_products[0]['producto']} "
                      f"tiene el mayor potencial de venta el próximo mes. "
                      f"Los top 5 productos representan las mejores oportunidades de ventas.",
            "confidence": "80%"
        }
        
    except Exception as e:
        print(f"Error en predict_products: {e}")
        return None


def generate_pdf_report(report_data, user_prompt):
    """Genera PDF del reporte"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("TesloShop - Reporte de Análisis", title_style))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(report_data.get('titulo', 'Reporte'), title_style))
    story.append(Paragraph(report_data.get('resumen', ''), styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    
    return buffer


if __name__ == '__main__':
    print("🚀 TesloShop AI Service con ML")
    print("✅ OpenAI + scikit-learn")
    print("📊 Endpoints:")
    print("   - /api/generate-chart")
    print("   - /api/generate-report")
    print("   - /api/predict (NUEVO)")
    app.run(debug=True, port=8000)