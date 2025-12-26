import io
import traceback
from datetime import datetime
from difflib import get_close_matches
import google.generativeai as genai
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests

# NLP & feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# anomaly detection
from sklearn.ensemble import IsolationForest

# Google Gemini API key
genai.configure(api_key= Gemini_API)  # 🔐 replace with your real key

# Forecasting: Prophet preferred, fallback to ARIMA
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

# --- Configurable thresholds ---
RELATIVE_DROP_THRESHOLD = 0.30
ROLLING_WINDOW = 7
ISOFORREST_CONTAMINATION = 0.05
MIN_ROWS_FOR_FORECAST = 30

STANDARD_COLS = {
    "date": ["date", "day", "timestamp", "datetime", "time"],
    "revenue": ["revenue", "rev", "sales", "sale", "amount", "total_revenue"],
    "ad_spend": ["ad_spend", "spend", "cost", "adcost", "ad_cost", "marketing_spend"],
    "clicks": ["clicks", "click", "cnt_clicks"],
    "impressions": ["impressions", "impression", "imps", "views"],
    "conversions": ["conversions", "conversion", "orders", "purchases", "leads"],
    "channel": ["channel", "source", "platform", "utm_source", "campaign"]
}

@app.route('/api/custom_chart', methods=['POST'])
def custom_chart():
    try:
        payload = request.get_json()
        print("\n=== /api/custom_chart DEBUG ===")
        print("Payload received:", payload)

        query = payload.get('query', '').lower()
        file_data = payload.get('data')

        # 🚨 Dataset presence check
        if not file_data:
            print("❌ No dataset found in payload")
            return jsonify({'error': 'Please analyze your dataset first'}), 400

        # ✅ Parse dataset
        if isinstance(file_data, list) and isinstance(file_data[0], dict):
            keys = list(file_data[0].keys())
            if len(keys) == 1 and '\t' in keys[0]:
                # Tab-separated fallback
                rows = [list(item.values())[0].split('\t') for item in file_data]
                headers = keys[0].split('\t')
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame(file_data)
        else:
            print("❌ Invalid data format")
            return jsonify({'error': 'Invalid dataset format received'}), 400

        # ✅ Clean and normalize column names
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # ✅ Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        print(f"✅ Parsed columns: {df.columns.tolist()}")

        # === Column Mapping ===
        def map_columns(columns):
            mapping = {}
            for c in columns:
                if 'date' in c or 'time' in c:
                    mapping['date'] = c
                if 'channel' in c or 'source' in c or 'medium' in c:
                    mapping['channel'] = c
                if 'revenue' in c or 'sales' in c or 'profit' in c:
                    mapping['revenue'] = c
                if 'spend' in c or 'cost' in c or 'budget' in c:
                    mapping['ad_spend'] = c
                if 'click' in c or 'engagement' in c or 'view' in c:
                    mapping['clicks'] = c
                if 'conversion' in c or 'lead' in c or 'signup' in c:
                    mapping['conversions'] = c
                if 'roi' in c:
                    mapping['roi'] = c
            return mapping

        mapping = map_columns(df.columns)
        chart_type = 'bar'
        x, y = None, None

        # === Query Interpretation ===
        if 'roi' in query:
            x, y = mapping.get('channel'), mapping.get('roi') or mapping.get('revenue')
            chart_type = 'bar'
        elif 'conversion' in query:
            x, y = mapping.get('channel'), mapping.get('conversions')
            chart_type = 'bar'
        elif 'spend' in query and 'sales' in query:
            x, y = mapping.get('ad_spend'), mapping.get('revenue')
            chart_type = 'scatter'
        elif 'engagement' in query or 'click' in query:
            x, y = mapping.get('date'), mapping.get('clicks')
            chart_type = 'line'
        elif 'sales' in query and 'time' in query:
            x, y = mapping.get('date'), mapping.get('revenue')
            chart_type = 'line'

        # === Fallback: choose first 2 numeric columns if mapping failed ===
        if not x or not y:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x, y = numeric_cols[0], numeric_cols[1]
                print(f"⚠️ Fallback to numeric columns: {x}, {y}")
            else:
                print("❌ Could not determine suitable columns")
                return jsonify({'error': 'Could not determine suitable columns for charting'}), 400

        # ✅ Prepare chart data
        data = df[[x, y]].dropna()
        labels = data[x].astype(str).tolist()[:50]
        values = pd.to_numeric(data[y], errors='coerce').fillna(0).tolist()[:50]

        print(f"📊 Chart ready — type={chart_type}, X={x}, Y={y}, Points={len(labels)}")

        return jsonify({'labels': labels, 'values': values, 'type': chart_type}), 200

    except Exception as e:
        traceback.print_exc()
        print("❌ Exception:", e)
        return jsonify({'error': str(e)}), 500


# --- Helper: column mapping ---
def map_columns(df_columns):
    lowered = [c.lower().strip() for c in df_columns]
    mapping = {}
    for std, synonyms in STANDARD_COLS.items():
        match = None
        for syn in synonyms:
            candidates = get_close_matches(syn, lowered, n=1, cutoff=0.6)
            if candidates:
                match = candidates[0]
                break
        if not match:
            for c in lowered:
                for syn in synonyms:
                    if syn in c:
                        match = c
                        break
                if match:
                    break
        if match:
            orig = next((c for c in df_columns if c.lower() == match), match)
            mapping[std] = orig
        else:
            mapping[std] = None
    return mapping

# --- File Readers ---
def read_csv_file(file_storage):
    content = file_storage.read()
    try:
        # Automatically detect delimiter (comma, tab, semicolon, etc.)
        return pd.read_csv(io.BytesIO(content), sep=None, engine='python')
    except Exception:
        return pd.read_csv(io.BytesIO(content), encoding='utf-8', engine='python', sep=None)


def read_csv_from_sheet_url(sheet_url):
    if 'docs.google.com' in sheet_url and 'export' not in sheet_url:
        gid = '0'
        if 'gid=' in sheet_url:
            gid = sheet_url.split('gid=')[1].split('&')[0]
        sid = sheet_url.split('/d/')[1].split('/')[0]
        sheet_url = f'https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}'
    r = requests.get(sheet_url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# --- Detect date column ---
def detect_and_parse_date(df):
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'day'])]
    if not date_cols:
        return None, df
    chosen = next((c for c in date_cols if c.lower() == 'date'), date_cols[0])
    try:
        df[chosen] = pd.to_datetime(df[chosen], errors='coerce')
        df = df.dropna(subset=[chosen]).sort_values(chosen).reset_index(drop=True)
        return chosen, df
    except Exception:
        return None, df

# --- Compute Marketing Metrics ---
def compute_metrics(df, mapping):
    def safe_sum(col):
        if col and col in df.columns:
            try:
                return float(df[col].astype(float).sum())
            except Exception:
                return None
        return None

    revenue = safe_sum(mapping.get('revenue'))
    spend = safe_sum(mapping.get('ad_spend'))
    conversions = safe_sum(mapping.get('conversions'))
    clicks = safe_sum(mapping.get('clicks'))
    impressions = safe_sum(mapping.get('impressions'))

    res = {
        'roi_pct': round(((revenue - spend) / spend) * 100, 2) if revenue and spend else None,
        'conversion_rate_pct': (
            round((conversions / impressions) * 100, 3)
            if conversions and impressions else (round((conversions / clicks) * 100, 3)
            if conversions and clicks else None)
        ),
        'cpc': round(spend / clicks, 3) if spend and clicks else None,
        'engagement_rate': round((clicks / impressions) * 100, 3) if clicks and impressions else None,
        'used_columns': mapping
    }
    return res

# --- Prophet / ARIMA Forecast ---
def forecast_series(ts_series, steps=30):
    try:
        ts_series = ts_series.dropna()

        if ts_series.shape[0] < MIN_ROWS_FOR_FORECAST:
            return None

        # --- Ensure datetime index is valid and sorted ---
        ts_series.index = pd.to_datetime(ts_series.index, format='%d-%m-%Y', errors='coerce')
        ts_series = ts_series.sort_index().dropna()

        if not np.issubdtype(ts_series.index.dtype, np.datetime64):
            return None

        # Prepare DataFrame for Prophet
        prophet_df = pd.DataFrame({
            'ds': ts_series.index,
            'y': ts_series.values.astype(float)
        }).reset_index(drop=True)

        if PROPHET_AVAILABLE:
            from prophet import Prophet
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            model.fit(prophet_df)

            # Create future dates
            future = model.make_future_dataframe(periods=steps, freq='D')
            forecast = model.predict(future)

            # Extract forecast
            future_preds = forecast.tail(steps)
            return {
                'dates': future_preds['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'values': future_preds['yhat'].astype(float).round(2).tolist()
            }

        # --- Fallback to ARIMA ---
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(ts_series.astype(float), order=(1, 1, 1)).fit()
        preds = model.forecast(steps=steps)
        last_date = ts_series.index.max()
        future_dates = [(last_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(steps)]
        return {'dates': future_dates, 'values': [float(x) for x in preds]}

    except Exception as e:
        print("Forecast error:", e)
        traceback.print_exc()
        return None



# --- Analyze Text Description ---
def analyze_description(text):
    if not text or not text.strip():
        return {'keywords': [], 'intent': None, 'target_audience': None, 'marketing_channels': []}
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        top_idx = scores.argsort()[::-1][:8]
        keywords = [feature_names[i] for i in top_idx]
    except Exception:
        tokens = [w.lower() for w in text.split() if len(w) > 2]
        freq = {t: tokens.count(t) for t in set(tokens)}
        keywords = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:8]
    lowered = text.lower()
    intent = None
    if any(k in lowered for k in ['acquire', 'lead', 'sales', 'growth', 'revenue']):
        intent = 'growth / lead generation'
    elif any(k in lowered for k in ['brand', 'awareness']):
        intent = 'brand awareness'
    elif any(k in lowered for k in ['retention', 'churn']):
        intent = 'retention'
    channels = [ch for ch in ['google ads', 'facebook', 'instagram', 'linkedin', 'tiktok', 'email', 'organic', 'seo', 'youtube'] if ch in lowered]
    aud = 'B2B' if 'b2b' in lowered else ('B2C' if 'b2c' in lowered else None)
    return {'keywords': keywords, 'intent': intent, 'target_audience': aud, 'marketing_channels_mentioned': channels}

# --- Chart Prep ---
def build_chart_data(series):
    use = series.loc[series.index >= series.index.max() - pd.Timedelta(days=90)] if np.issubdtype(series.index.dtype, np.datetime64) else series.iloc[-90:]
    labels = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in use.index]
    values = [float(v) if pd.notna(v) else 0.0 for v in use.values]
    return labels, values

# --- AI Insight Generator (Gemini) ---
def generate_ai_insights(metrics, desc_analysis):
    try:
        prompt = f"""
        You are a marketing data analyst.
        Analyze the following metrics and provide 3 key findings, 2 opportunities, and 2 alerts.
        Return the result as a valid JSON (no explanations) in this exact format:
        {{
          "findings": ["..."],
          "opportunities": ["..."],
          "alerts": ["..."]
        }}
        Metrics: {metrics}
        Business Context: {desc_analysis}
        """
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)

        # Clean up response text
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
            return json.dumps(data)  # return proper JSON string
        except Exception:
            return text  # fallback
    except Exception as e:
        print("AI Insight Error:", e)
        return None


# --- Main API ---
@app.route('/api/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        description = ''
        df = None

        if 'multipart/form-data' in (request.content_type or ''):
            description = request.form.get('description', '')
            if 'file' in request.files and request.files['file'].filename:
                df = read_csv_file(request.files['file'])
            elif request.form.get('sheet_url'):
                df = read_csv_from_sheet_url(request.form.get('sheet_url'))
        else:
            payload = request.get_json() or {}
            description = payload.get('description', '')
            if payload.get('sheet_url'):
                df = read_csv_from_sheet_url(payload['sheet_url'])

        desc_analysis = analyze_description(description)
        if df is None:
            return jsonify({'status': 'ok', 'description_analysis': desc_analysis, 'message': 'No dataset provided'}), 200

        df.columns = [c.strip() for c in df.columns]
        mapping = map_columns(df.columns)
        date_col, df = detect_and_parse_date(df)
        if date_col:
            df = df.set_index(date_col)

        metrics = compute_metrics(df, mapping)

        # Select main metric for visualization
        main_metric = next((mapping[k] for k in ['revenue', 'conversions', 'clicks'] if mapping[k]), None)
        if not main_metric:
            main_metric = df.select_dtypes(include=[np.number]).columns[0]

        ts = df[main_metric].astype(float)
        if date_col:
            ts = ts.resample('W').sum()

        labels, values = build_chart_data(ts)

        forecast_res = forecast_series(ts, 30)
        if not forecast_res:
            forecast_res = {"dates": [], "values": []}


        ai_insights = generate_ai_insights(metrics, desc_analysis)

        return jsonify({
            'status': 'ok',
            'metrics': metrics,
            'description_analysis': desc_analysis,
            'chart': {
                'labels': labels,
                'values': values,
                'forecast': forecast_res
            },
            'ai_insights': ai_insights
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
