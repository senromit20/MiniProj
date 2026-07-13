# CarMatch — Car Recommendation Web App

A Flask-based Car recommendation system powered by a Random Forest + Cosine Similarity model, trained on 2,564 Indian cars across 15 brands.

This was our KIIT 6th semester Minor Project under the guidance of Mr. Murari Mandal.

## Project Structure

```
carapp/
├── app.py                    ← Flask backend
├── car_dataset_cleaned.csv   ← Cleaned dataset
├── requirements.txt
├── templates/
│   └── index.html            ← Frontend (single-page form)
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

## How It Works

The user answers 8 questions on the website:
1. **Budget** — selects a price range (Budget / Value / Mid-Range / Premium / Luxury)
2. **Fuel type** — Petrol, Diesel, CNG, Electric, Hybrid, or no preference
3. **Vehicle type** — SUV, Sedan, Hatchback, MPV, Sports, or no preference
4. **Transmission** — Manual, Automatic, AMT/DCT, or no preference
5. **Seating capacity** — minimum seats required
6. **Brand preference** — any of the 15 brands or no preference
7. **Service budget** — annual maintenance budget (Low / Moderate / High)
8. **Priorities** — sliders for Safety, Comfort, Technology, Performance

The backend applies hard filters (budget, fuel, type, seating, service cost, brand) and then runs:
- **Random Forest Classifier** → predicts the single best car model
- **Cosine Similarity** → ranks all filtered candidates by how well their scores match user priorities
- **Combined scoring** → 60% similarity score + 40% price proximity

Returns Top 5 recommendations with match percentage, scores, specs, and price range.

## Model Performance
- Training accuracy: 100% (test set)
- 5-Fold CV Accuracy: 99.84% ± 0.08%
- 130 car models across 15 brands
