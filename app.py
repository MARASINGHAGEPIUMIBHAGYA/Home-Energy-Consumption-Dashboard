from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

SEQ_LENGTH = 24

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_str = request.form.get("input_values", "")
        try:
            values = [float(x.strip()) for x in input_str.split(",") if x.strip() != '']
            if len(values) != SEQ_LENGTH:
                raise ValueError(f"Please enter exactly {SEQ_LENGTH} numeric values.")
            
            arr = np.array(values).reshape(-1, 1)
            scaled = scaler.transform(arr).flatten().reshape(1, -1)  # shape (1, 24)
            
            pred_scaled = model.predict(scaled)
            
            pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0][0]
            
            return render_template("dashboard.html", forecast=round(pred, 3))
        except Exception as e:
            return render_template("dashboard.html", error=str(e))
    return render_template("dashboard.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic PORT
    app.run(host="0.0.0.0", port=port, debug=True)
