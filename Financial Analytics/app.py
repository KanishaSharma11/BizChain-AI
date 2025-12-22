from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import traceback
import re
from prophet import Prophet
from dotenv import load_dotenv
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

FUTURE_PERIODS = 4
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyAcVz99F3MIfSf29ns8OK07qMVpwXRuTVg"))

@app.route('/')
def home():
    return render_template('FinancialAnalytics.html')


# ------------------------- Utilities -------------------------
def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def clean_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object or series.dtype == "string":
        s = series.astype(str).str.strip()
        s = s.str.replace(r'[\(\)\$\€\£]', '', regex=True)
        s = s.str.replace(r'[^\d\-,\.]', '', regex=True)
        s = s.str.replace(',', '', regex=True)
        s = s.replace('', np.nan)
        return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

def parse_quarter_string_to_date(s: str):
    if not isinstance(s, str):
        return pd.NaT
    s = s.strip().upper()
    m = re.match(r'(\d{4})[^\dA-Z]*(Q?)([1-4])', s)
    if m:
        year = int(m.group(1))
        q = int(m.group(3))
        month = q * 3
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return pd.NaT

def parse_dates_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().mean() > 0.25:
        alt = series.astype(str).apply(parse_quarter_string_to_date)
        parsed = parsed.fillna(alt)
    return parsed

def safe_div(a, b):
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((b_arr == 0) | np.isnan(a_arr) | np.isnan(b_arr), np.nan, a_arr / b_arr)


# ------------------------- Prophet Forecast -------------------------
def forecast_with_prophet(df: pd.DataFrame, date_col: str, value_col: str, future_periods: int = FUTURE_PERIODS):
    work = df[[date_col, value_col]].copy()
    work.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True)
    work['ds'] = parse_dates_series(work['ds'])
    work['y'] = pd.to_numeric(work['y'], errors='coerce')
    work.dropna(subset=['ds', 'y'], inplace=True)
    work.sort_values('ds', inplace=True)

    if len(work) < 3:
        return work['ds'].dt.strftime('%Y-%m-%d').tolist(), work['y'].tolist(), [False]*len(work)

    try:
        model = Prophet()
        model.fit(work)
        freq = pd.infer_freq(work['ds']) or 'Q'
        future = model.make_future_dataframe(periods=future_periods, freq=freq)
        forecast = model.predict(future)
        forecast['is_forecast'] = forecast['ds'] > work['ds'].max()
        forecast['value'] = forecast['yhat']
        labels = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        values = forecast['value'].round(2).tolist()
        is_forecast = forecast['is_forecast'].tolist()
        return labels, values, is_forecast
    except Exception as e:
        print("Prophet error:", e)
        labels = work['ds'].dt.strftime('%Y-%m-%d').tolist()
        values = work['y'].tolist()
        flags = [False]*len(values)
        return labels, values, flags


# ------------------------- Main Analysis -------------------------
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400

        df = pd.read_csv(file)
        df.columns = [clean_column_name(c) for c in df.columns]

        mapping = {
            "date": next((c for c in df.columns if 'date' in c or 'period' in c or 'quarter' in c), None),
            "revenue": next((c for c in df.columns if 'revenue' in c or 'sales' in c), None),
            "net_income": next((c for c in df.columns if 'net_income' in c or 'profit' in c), None),
            "ebitda": next((c for c in df.columns if 'ebitda' in c), None),
            "total_assets": next((c for c in df.columns if 'asset' in c), None),
            "total_liabilities": next((c for c in df.columns if 'liabilit' in c), None),
            "shareholders_equity": next((c for c in df.columns if 'equity' in c or 'net_worth' in c), None),
            "current_assets": next((c for c in df.columns if 'current_assets' in c), None),
            "current_liabilities": next((c for c in df.columns if 'current_liabilities' in c), None),
        }

        required = ["revenue", "net_income", "total_assets", "shareholders_equity"]
        missing = [k for k in required if not mapping.get(k)]
        if missing:
            return jsonify({'error': f"Missing columns: {', '.join(missing)}"}), 400

        for col in df.columns:
            if col != mapping["date"]:
                df[col] = clean_numeric(df[col])

        df[mapping["date"]] = parse_dates_series(df[mapping["date"]])
        df.sort_values(by=mapping["date"], inplace=True)

        # Compute Metrics
        df['roe_pct'] = safe_div(df[mapping["net_income"]], df[mapping["shareholders_equity"]]) * 100
        df['roa_pct'] = safe_div(df[mapping["net_income"]], df[mapping["total_assets"]]) * 100
        df['net_profit_margin_pct'] = safe_div(df[mapping["net_income"]], df[mapping["revenue"]]) * 100
        df['current_ratio'] = safe_div(df.get(mapping["current_assets"], np.nan), df.get(mapping["current_liabilities"], np.nan))
        df['debt_to_equity'] = safe_div(df.get(mapping["total_liabilities"], np.nan), df.get(mapping["shareholders_equity"], np.nan))
        df['ebitda_margin_pct'] = safe_div(df.get(mapping["ebitda"], np.nan), df[mapping["revenue"]]) * 100

        latest = df.iloc[-1]
        metrics = {k: (round(v, 2) if not pd.isna(v) else None) for k, v in {
            "Return on Equity (%)": latest['roe_pct'],
            "Return on Assets (%)": latest['roa_pct'],
            "Net Profit Margin (%)": latest['net_profit_margin_pct'],
            "Current Ratio": latest['current_ratio'],
            "Debt to Equity": latest['debt_to_equity'],
            "EBITDA Margin (%)": latest['ebitda_margin_pct']
        }.items()}

        rev_labels, rev_values, rev_flags = forecast_with_prophet(df, mapping["date"], mapping["revenue"], FUTURE_PERIODS)
        chart = {"labels": rev_labels, "values": rev_values, "forecast_flags": rev_flags}

        return jsonify({
            "metrics": metrics,
            "chart": chart,
            "raw_data": df.fillna('').to_dict(orient='records'),
            "column_mapping": mapping
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ------------------------- Custom Chart -------------------------
@app.route('/api/custom_chart', methods=['POST'])
def custom_chart():
    try:
        content = request.get_json()
        query = content.get("query", "").lower()
        data = content.get("data", [])

        if not data:
            return jsonify({"error": "No dataset provided."}), 400

        df = pd.DataFrame(data)
        df.columns = [clean_column_name(c) for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c or 'quarter' in c or 'period' in c), None)
        if date_col:
            df[date_col] = parse_dates_series(df[date_col])
            df.sort_values(by=date_col, inplace=True)
        for col in df.columns:
            if col != date_col:
                df[col] = clean_numeric(df[col])

        # Detect chart type
        value_col = None
        chart_title = "Custom Chart"

        if "revenue" in query:
            value_col = next((c for c in df.columns if 'revenue' in c), None)
            chart_title = "Revenue Trend"
        elif "profit" in query:
            value_col = next((c for c in df.columns if 'net_income' in c or 'profit' in c), None)
            chart_title = "Profit Trend"

        if not value_col:
            value_col = df.select_dtypes(include=[np.number]).columns[0]

        labels = df[date_col].dt.strftime('%Y-%m-%d').fillna('').tolist() if date_col else list(range(len(df)))
        values = df[value_col].fillna(0).round(2).tolist()

        return jsonify({
            "title": chart_title,
            "type": "line",
            "labels": labels,
            "values": values
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ------------------------- AI Insights -------------------------
@app.route('/api/ai_insights', methods=['POST'])
def ai_insights():
    try:
        content = request.get_json()
        company_description = content.get("companyDescription", "").strip()
        analysis_data = content.get("analysis", {})

        if not company_description:
            return jsonify({"error": "Company description is required."}), 400
        if not analysis_data:
            return jsonify({"error": "Financial analysis data is required."}), 400

        prompt = f"""
You are a financial strategy expert.
Analyze the following company's financial data.

Company Description:
{company_description}

Financial Data:
{analysis_data}

Please return a clean JSON (no code blocks, no markdown) with exactly 3 short bullet points per category:

{{
  "keyFindings": ["point1", "point2", "point3"],
  "opportunities": ["point1", "point2", "point3"],
  "suggestions": ["point1", "point2", "point3"]
}}
"""

        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)

        text = response.text.strip()

        # Remove markdown formatting if present
        text = text.replace("```json", "").replace("```", "").strip()

        import json
        try:
            parsed = json.loads(text)
        except Exception:
            # fallback: try extracting JSON substring
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except:
                    parsed = {}
            else:
                parsed = {}

        # Ensure proper format (lists for all fields)
        def ensure_list(val):
            if isinstance(val, list):
                return val[:3]
            elif isinstance(val, str):
                parts = [p.strip("-• ") for p in val.split("\n") if p.strip()]
                return parts[:3] if parts else [val]
            else:
                return ["No data"]

        parsed = {
            "keyFindings": ensure_list(parsed.get("keyFindings", [])),
            "opportunities": ensure_list(parsed.get("opportunities", [])),
            "suggestions": ensure_list(parsed.get("suggestions", []))
        }

        return jsonify(parsed), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"AI Insights generation failed: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(debug=True, port=5002)
