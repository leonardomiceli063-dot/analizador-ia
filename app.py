from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import tempfile

app = Flask(__name__, static_folder='static')
CORS(app)

# Estado simple (temporal)
estado = {}

# =========================
# ANALIZADOR IA
# =========================
class AnalizadorIA:
    def __init__(self, df, col_valor, col_categoria, col_fecha):
        self.df = df.copy()
        self.col_valor = col_valor
        self.col_categoria = col_categoria
        self.col_fecha = col_fecha
        self.anomalias_df = None

    def detectar_anomalias(self):
        columnas_numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if columnas_numericas:
            datos = self.df[columnas_numericas].fillna(self.df[columnas_numericas].mean())

            scaler = StandardScaler()
            datos_escalados = scaler.fit_transform(datos)

            modelo = IsolationForest(contamination=0.1, random_state=42)
            pred = modelo.fit_predict(datos_escalados)

            self.df['anomalia'] = np.where(pred == -1, 'Anomalía', 'Normal')
            self.anomalias_df = self.df[self.df['anomalia'] == 'Anomalía']

            return self.anomalias_df

        return pd.DataFrame()

    def analizar_tendencia(self):
        df = self.df.groupby(self.col_fecha)[self.col_valor].sum()

        if len(df) < 2:
            return "Datos insuficientes"

        x = np.arange(len(df))
        y = df.values.astype(float)

        pendiente = np.polyfit(x, y, 1)[0]

        if pendiente > 0:
            return "ALCISTA"
        elif pendiente < 0:
            return "BAJISTA"
        return "ESTABLE"

# =========================
# RUTAS
# =========================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']

    tmp = tempfile.NamedTemporaryFile(delete=False)
    file.save(tmp.name)

    df = pd.read_csv(tmp.name) if file.filename.endswith('.csv') else pd.read_excel(tmp.name)

    estado['df'] = df

    return jsonify({
        'columnas': df.columns.tolist(),
        'preview': df.head(5).to_dict(orient='records')
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json

    df = estado.get('df')

    if df is None:
        return jsonify({'error': 'No data'}), 400

    col_valor = data['col_valor']
    col_fecha = data['col_fecha']
    col_categoria = data.get('col_categoria')

    df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
    df = df.dropna(subset=[col_fecha])

    analizador = AnalizadorIA(df, col_valor, col_categoria, col_fecha)

    anomalias = analizador.detectar_anomalias()

    return jsonify({
        'total': len(df),
        'anomalias': anomalias.head(20).to_dict(orient='records'),
        'tendencia': analizador.analizar_tendencia()
    })


@app.route('/api/export', methods=['GET'])
def export():
    df = estado.get('df')

    if df is None:
        return jsonify({'error': 'No data'}), 400

    csv = df.to_csv(index=False)

    return Response(csv, mimetype='text/csv')


# =========================
# RUN
# =========================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=port)