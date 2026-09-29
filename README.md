# Student Productivity Predictor (Flask TP)

Predicts `productivity_score` (0 to 100) from 6 student habits.
Best model: Linear Regression (R² ≈ 1.000, RMSE ≈ 0.003).

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
python train.py                  # optional: rebuilds model.pkl with your scikit-learn version
python app.py                    # open http://127.0.0.1:5000/
```

## Deploy on Heroku
```bash
heroku login
heroku create productivity-predictor-yourname --region eu
git init
git add .
git commit -am "initial change"
git push heroku master           # use "main" if your branch is called main
heroku open
```
If the build fails: `heroku logs --tail`.
