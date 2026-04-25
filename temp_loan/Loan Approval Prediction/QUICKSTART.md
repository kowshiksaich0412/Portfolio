# ⚡ Quick Start Guide

## 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Train the Model
```bash
python train.py
```
This will:
- Load the dataset
- Preprocess and analyze data
- Train 3 ML models
- Save the best model to `models/`
- Generate visualization plots

**Time**: ~10-30 seconds

## 3️⃣ Run the Web App
```bash
python app.py
```

## 4️⃣ Access the Application
Open your browser: **http://localhost:5000**

---

## 📋 What You Get

### After Training (`train.py`):
```
data/
  └── loan_data.csv                    (Sample dataset provided)

models/
  ├── best_model.pkl                   (Trained model)
  ├── scaler.pkl                       (Feature scaler)
  └── label_encoders.pkl               (Categorical encoders)

static/plots/
  ├── class_distribution.png
  ├── correlation_heatmap.png
  ├── feature_distributions.png
  ├── confusion_matrices.png
  └── roc_curves.png
```

### After Running Web App (`app.py`):
- Interactive prediction interface at `http://localhost:5000`
- REST API at `/predict` endpoint
- Beautiful, responsive UI

---

## 🎯 Example Prediction

### Via Web Interface:
1. Open http://localhost:5000
2. Fill in applicant details
3. Click "Get Prediction"
4. View results with confidence score

### Via API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "credit_score": 720,
    "employment_length": 10,
    "loan_amount": 250000,
    "interest_rate": 4.5,
    "loan_type": "Personal",
    "employment_type": "Employed"
  }'
```

---

## 🛑 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: best_model.pkl` | Run `python train.py` first |
| `Port 5000 already in use` | Kill process or change port in app.py |
| Prediction endpoint returns error | Check input JSON format matches API spec |

---

## 📊 Project Files

| File | Purpose |
|------|---------|
| `train.py` | Data processing, model training, evaluation |
| `app.py` | Flask web server and REST API |
| `templates/index.html` | Web interface |
| `data/loan_data.csv` | Sample dataset |
| `requirements.txt` | Python dependencies |
| `README.md` | Complete documentation |

---

## 🚀 Next Steps

1. **Customize Dataset**: Replace `data/loan_data.csv` with your own data
2. **Tune Parameters**: Adjust hyperparameters in `train.py`
3. **Deploy**: Use Docker or cloud platforms (AWS, Heroku, etc.)
4. **Integrate**: Connect to databases or other services
5. **Monitor**: Add logging and performance monitoring

---

For more details, see **README.md**
