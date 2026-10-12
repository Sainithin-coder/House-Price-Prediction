"""
Flask Backend for House Price Prediction Web Application.
Endpoints:
  GET  /              → Serve main page
  POST /predict       → Return predicted price + breakdown
  GET  /stats         → Dataset statistics for dashboard charts
  GET  /history       → Training history
  POST /train         → Trigger model retraining
"""
import os, sys, json, traceback
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from models.joint_attention_model import build_model
from utils.feature_engineering import CATEGORICAL_COLS, NUMERICAL_COLS

app = Flask(__name__)
CORS(app)

# ─── Globals ──────────────────────────────────────────────────────────────────
MODEL = None
FE = None
META = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── Loader ───────────────────────────────────────────────────────────────────
def load_model():
    global MODEL, FE, META
    try:
        with open('models/model_meta.json') as f:
            META = json.load(f)
        FE = joblib.load('models/feature_engineer.pkl')
        MODEL = build_model(
            n_num_features=META['n_num_features'],
            cat_dims=META['cat_dims'],
            device=DEVICE
        )
        MODEL.load_state_dict(torch.load('models/best_model.pt', map_location=DEVICE))
        MODEL.eval()
        print(f"[App] Model loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f"[App] Model load failed: {e}")
        return False


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL, FE
    if MODEL is None:
        loaded = load_model()
        if not loaded:
            return jsonify({'error': 'Model not trained yet. Please run training first.'}), 503

    try:
        data = request.get_json(force=True)

        # Build a single-row DataFrame
        row = {
            'city':            data.get('city', 'Maharashtra'),
            'neighborhood':    data.get('neighborhood', 'Baner'),
            'property_type':   data.get('property_type', 'Apartment'),
            'condition':       data.get('condition', 'Good'),
            'garage_type':     data.get('garage_type', 'Covered Parking'),
            'school_rating':   data.get('school_rating', 'B'),
            'sqft':            float(data.get('sqft', 1400)),
            'bedrooms':        int(data.get('bedrooms', 3)),
            'bathrooms':       int(data.get('bathrooms', 2)),
            'year_built':      int(data.get('year_built', 2010)),
            'lot_size':        float(data.get('lot_size', 2000)),
            'floors':          int(data.get('floors', 2)),
            'has_pool':        int(data.get('has_pool', 0)),
            'has_basement':    int(data.get('has_basement', 0)),
            'distance_cbd':    float(data.get('distance_cbd', 8)),
            'crime_rate':      float(data.get('crime_rate', 3)),
            'walk_score':      int(data.get('walk_score', 65)),
            'renovation_year': int(data.get('renovation_year', 2015)),
        }

        df = pd.DataFrame([row])
        X_num, X_cat = FE.transform(df)
        x_num_t = torch.tensor(X_num, dtype=torch.float32).to(DEVICE)
        x_cat_t = torch.tensor(X_cat, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            pred_scaled, attn_num, attn_cat = MODEL(x_num_t, x_cat_t)

        price = float(FE.inverse_transform_target(pred_scaled.cpu().numpy())[0])
        price = max(50000, price)

        # Feature importance via attention (average over last self-attn layer, all heads)
        num_attn = attn_num[-1][0].mean(0).mean(0).cpu().numpy().tolist()
        cat_attn = attn_cat[-1][0].mean(0).mean(0).cpu().numpy().tolist()

        feat_names = META.get('feature_names', [])
        cat_names  = list(META.get('cat_dims', {}).keys())

        num_importance = [
            {'feature': feat_names[i] if i < len(feat_names) else f'num_{i}', 'score': round(float(v), 4)}
            for i, v in enumerate(num_attn[:len(feat_names)])
        ]
        cat_importance = [
            {'feature': cat_names[i] if i < len(cat_names) else f'cat_{i}', 'score': round(float(v), 4)}
            for i, v in enumerate(cat_attn[:len(cat_names)])
        ]

        # Confidence interval (±8% heuristic from val MAE)
        low  = round(price * 0.92)
        high = round(price * 1.08)

        return jsonify({
            'predicted_price': round(price),
            'price_low':       low,
            'price_high':      high,
            'price_per_sqft':  round(price / row['sqft'], 2),
            'num_importance':  num_importance[:8],
            'cat_importance':  cat_importance,
            'model_info': {
                'architecture': 'Joint Self-Attention Network',
                'best_val_loss': META.get('best_val_loss'),
                'final_val_mae': META.get('final_val_mae'),
                'device': DEVICE
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    try:
        df = pd.read_csv('data/house_data.csv')
        # Price distribution buckets
        hist, edges = np.histogram(df['price'], bins=20)
        price_dist = {
            'bins': [f"${int(e/1000)}K" for e in edges[:-1]],
            'counts': hist.tolist()
        }
        # City avg price
        city_avg = df.groupby('city')['price'].mean().round(0).sort_values(ascending=False)
        # Property type avg
        type_avg = df.groupby('property_type')['price'].mean().round(0).sort_values(ascending=False)
        # Correlation of numerical features with price
        num_cols = ['sqft', 'bedrooms', 'bathrooms', 'lot_size', 'walk_score',
                    'crime_rate', 'distance_cbd', 'has_pool', 'has_basement']
        corr = df[num_cols + ['price']].corr()['price'].drop('price').round(3)

        return jsonify({
            'total_records': len(df),
            'avg_price': round(float(df['price'].mean())),
            'min_price': int(df['price'].min()),
            'max_price': int(df['price'].max()),
            'price_distribution': price_dist,
            'city_avg_price': {k: int(v) for k, v in city_avg.items()},
            'property_type_avg': {k: int(v) for k, v in type_avg.items()},
            'feature_correlations': corr.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def training_history():
    try:
        with open('models/training_history.json') as f:
            hist = json.load(f)
        return jsonify(hist)
    except:
        return jsonify({'error': 'No training history found'}), 404


@app.route('/train', methods=['POST'])
def retrain():
    try:
        from train import train
        history = train()
        load_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully', 'history': history})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'device': DEVICE
    })


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
