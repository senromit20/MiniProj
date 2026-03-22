"""
CarMatch — Flask Backend
"""
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle, re, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── Load model artefacts once ──
with open('car_model_artefacts.pkl', 'rb') as f:
    ARTS = pickle.load(f)

MODEL     = ARTS['model']
SCALER    = ARTS['scaler']
LABEL_ENC = ARTS['label_encoders']
TARGET_LE = ARTS['target_le']
FEAT_COLS = ARTS['feature_cols']
DF        = ARTS['df_clean'].copy()

# Ensure helper columns exist
def _parse_service(s):
    s = str(s)
    if s == 'Low':      return 1
    if s == 'Moderate': return 2
    if s == 'High':     return 3
    nums = re.findall(r'[\d]+', s.replace(',', ''))
    return int(np.mean([int(n) for n in nums])) if nums else 2

if 'SERVICE_COST_NUM' not in DF.columns:
    DF['SERVICE_COST_NUM'] = DF['Service Cost'].apply(_parse_service)
if 'TARGET' not in DF.columns:
    DF['TARGET'] = DF['BRAND'] + ' ' + DF['MODEL']

BUDGET_RANGES = {
    'budget':    (0,         600_000),
    'value':     (600_000,   1_500_000),
    'mid-range': (1_500_000, 3_000_000),
    'premium':   (3_000_000, 7_000_000),
    'luxury':    (7_000_000, 999_999_999),
}

SERVICE_BUDGET_RANGES = {
    'low':      (0,      8_000),
    'moderate': (8_000,  15_000),
    'high':     (15_000, 999_999),
}

def service_cost_matches(svc_val, budget_label):
    """
    Handle mixed service cost values:
    - Ordinal 1/2/3 (Low/Moderate/High text from dataset)
    - Actual rupee amounts (numeric ranges parsed from strings)
    """
    lo, hi = SERVICE_BUDGET_RANGES.get(budget_label, (0, 999_999))
    if svc_val <= 3:
        # Ordinal: 1=Low, 2=Moderate, 3=High — map to budget label
        mapping = {'low': [1], 'moderate': [1, 2], 'high': [1, 2, 3]}
        return svc_val in mapping.get(budget_label, [1, 2, 3])
    return lo <= svc_val <= hi

BRANDS = sorted(DF['BRAND'].unique().tolist())


def build_user_vector(budget_mid, fuel, vehicle_cat, transmission,
                      seating, safety_w, comfort_w, tech_w, perf_w):
    row = {
        'PRICE(INR)':          budget_mid,
        'FUEL_TYPE_CLEAN':     LABEL_ENC['FUEL_TYPE_CLEAN'].transform([fuel])[0],
        'VEHICLE_CATEGORY':    LABEL_ENC['VEHICLE_CATEGORY'].transform([vehicle_cat])[0],
        'TRANSMISSION_TYPE':   LABEL_ENC['TRANSMISSION_TYPE'].transform([transmission])[0],
        'SEATING CAPACITY':    seating,
        'SAFETY_SCORE':        safety_w,
        'COMFORT_SCORE':       comfort_w,
        'TECH_SCORE':          tech_w,
        'PERFORMANCE_SCORE':   perf_w,
        'HORSEPOWER':          float(DF['HORSEPOWER'].median()),
        'TORQUE(LB.FT)':       float(DF['TORQUE(LB.FT)'].median()),
        'DISPLACEMENT':        float(DF['DISPLACEMENT'].median()),
        'MILEAGE(km/L)':       float(DF['MILEAGE(km/L)'].median()),
        'AIRBAGS':             float(DF['AIRBAGS'].median()),
        'SERVICE_COST_NUM':    float(DF['SERVICE_COST_NUM'].median()),
        'NUMBER OF CYLINDERS': float(DF['NUMBER OF CYLINDERS'].median()),
        'CHILD_LOCK':          1,
        'LENGTH_MM':           float(DF['LENGTH_MM'].median()),
        'WIDTH_MM':            float(DF['WIDTH_MM'].median()),
        'HEIGHT_MM':           float(DF['HEIGHT_MM'].median()),
    }
    vec = pd.DataFrame([row])[FEAT_COLS]
    return SCALER.transform(vec)


def recommend(budget_label, fuel, vehicle_type, transmission, seating,
              safety_p, comfort_p, tech_p, perf_p,
              brand_pref, service_budget_label, top_n=5):

    budget_lo, budget_hi = BUDGET_RANGES.get(budget_label, (0, 999_999_999))
    budget_mid = (budget_lo + budget_hi) / 2


    # ── Step 1: Hard filters ──
    cands = DF.copy()
    cands = cands[(cands['PRICE(INR)'] >= budget_lo) & (cands['PRICE(INR)'] <= budget_hi)]

    if fuel != 'any':
        cands = cands[cands['FUEL_TYPE_CLEAN'] == fuel]
    if vehicle_type != 'any':
        cands = cands[cands['VEHICLE_CATEGORY'] == vehicle_type]
    if transmission != 'any':
        cands = cands[cands['TRANSMISSION_TYPE'] == transmission]
    if seating > 0:
        cands = cands[cands['SEATING CAPACITY'] >= seating]

    # Service cost filter (handles mixed ordinal 1/2/3 and rupee values)
    cands = cands[cands['SERVICE_COST_NUM'].apply(
        lambda v: service_cost_matches(v, service_budget_label)
    )]

    # Brand preference filter (soft — try strict first)
    if brand_pref != 'any':
        brand_filtered = cands[cands['BRAND'] == brand_pref]
        if len(brand_filtered) >= 3:
            cands = brand_filtered

    # Relax filters if too few candidates
    if len(cands) < 3:
        cands = DF.copy()
        cands = cands[(cands['PRICE(INR)'] >= budget_lo * 0.8) & (cands['PRICE(INR)'] <= budget_hi * 1.2)]

    # ── Step 2: Classifier prediction ──
    fuel_enc  = fuel  if fuel  != 'any' else 'Petrol'
    vtype_enc = vehicle_type if vehicle_type != 'any' else 'SUV'
    trans_enc = transmission if transmission != 'any' else 'Manual/Auto'

    user_vec = build_user_vector(budget_mid, fuel_enc, vtype_enc, trans_enc,
                                 seating, safety_p, comfort_p, tech_p, perf_p)
    pred_class = MODEL.predict(user_vec)[0]
    pred_model = TARGET_LE.inverse_transform([pred_class])[0]

    # ── Step 3: Cosine similarity scoring ──
    score_cols = ['SAFETY_SCORE', 'COMFORT_SCORE', 'TECH_SCORE', 'PERFORMANCE_SCORE']
    weights = np.array([safety_p, comfort_p, tech_p, perf_p], dtype=float)
    weights /= weights.sum()
    user_scores = weights * np.array([safety_p, comfort_p, tech_p, perf_p])

    cands = cands.copy()
    cands_scores = cands[score_cols].values * weights
    sims = cosine_similarity([user_scores], cands_scores)[0]
    cands['_sim'] = sims

    # Cap budget_mid at the actual max price in candidates to avoid negative scores
    cand_max_price = cands['PRICE(INR)'].max() or 1
    cand_min_price = cands['PRICE(INR)'].min()
    budget_mid_capped = min(budget_mid, cand_max_price)
    price_range = max(cand_max_price - cand_min_price, 1)
    cands['_price_score'] = 1 - abs(cands['PRICE(INR)'] - budget_mid_capped) / price_range
    cands['_price_score'] = cands['_price_score'].clip(0, 1)
    cands['_final'] = 0.6 * cands['_sim'] + 0.4 * cands['_price_score']

    # ── Step 4: Aggregate to model level ──
    grp = (cands.groupby('TARGET')
           .agg(
               brand=('BRAND', 'first'),
               model=('MODEL', 'first'),
               vehicle_category=('VEHICLE_CATEGORY', 'first'),
               fuel_type=('FUEL_TYPE_CLEAN', 'first'),
               transmission=('TRANSMISSION_TYPE', 'first'),
               price_min=('PRICE(INR)', 'min'),
               price_max=('PRICE(INR)', 'max'),
               mileage=('MILEAGE(km/L)', 'mean'),
               safety=('SAFETY_SCORE', 'max'),
               comfort=('COMFORT_SCORE', 'max'),
               tech=('TECH_SCORE', 'max'),
               performance=('PERFORMANCE_SCORE', 'max'),
               airbags=('AIRBAGS', 'max'),
               seating=('SEATING CAPACITY', 'first'),
               score=('_final', 'mean'),
           )
           .reset_index()
           .sort_values('score', ascending=False))

    # Boost RF prediction to top
    if pred_model in grp['TARGET'].values:
        top = grp[grp['TARGET'] == pred_model]
        rest = grp[grp['TARGET'] != pred_model]
        grp = pd.concat([top, rest], ignore_index=True)

    results = []
    for _, row in grp.head(top_n).iterrows():
        results.append({
            'target':           row['TARGET'],
            'brand':            row['brand'],
            'model':            row['model'],
            'vehicle_category': row['vehicle_category'],
            'fuel_type':        row['fuel_type'],
            'transmission':     row['transmission'],
            'price_min':        int(row['price_min']),
            'price_max':        int(row['price_max']),
            'mileage':          round(float(row['mileage']), 1),
            'safety':           int(row['safety']),
            'comfort':          int(row['comfort']),
            'tech':             int(row['tech']),
            'performance':      int(row['performance']),
            'airbags':          int(row['airbags']),
            'seating':          int(row['seating']),
            'score':            round(float(row['score']) * 100, 1),
        })

    return results, pred_model


# ── Routes ──

@app.route('/')
def index():
    return render_template('index.html', brands=BRANDS)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    try:
        results, top_pick = recommend(
            budget_label        = data.get('budget', 'value'),
            fuel                = data.get('fuel', 'any'),
            vehicle_type        = data.get('vehicle_type', 'any'),
            transmission        = data.get('transmission', 'any'),
            seating             = int(data.get('seating', 5)),
            safety_p            = int(data.get('safety', 3)),
            comfort_p           = int(data.get('comfort', 3)),
            tech_p              = int(data.get('tech', 3)),
            perf_p              = int(data.get('performance', 3)),
            brand_pref          = data.get('brand', 'any'),
            service_budget_label= data.get('service_budget', 'moderate'),
            top_n               = 5,
        )
        return jsonify({'success': True, 'results': results, 'top_pick': top_pick})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
