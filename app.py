from flask import Flask, request, render_template_string
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Health Status Checker with AI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      color: #4b4b4b;
      margin: 0;
      padding-bottom: 60px; /* space for fixed footer */
    }
    .container {
      padding-bottom: 60px; /* prevent footer overlap */
    }
    .card {
      border-radius: 20px;
      box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    h1 {
      color: #4c51bf;
      font-weight: 700;
    }
    .mt-4 {
      margin-top: 2rem;
      padding: 1rem;
      border-radius: 12px;
      background-color: #f7f9fc;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
  </style>
</head>
<body>
  <div class="container py-5 d-flex justify-content-center">
    <div class="card p-4 bg-white" style="max-width: 480px; width: 100%;">
      <h1 class="mb-4 text-center">MyFit AI</h1>
      <form method="POST" novalidate>
        <div class="mb-3">
          <label for="weight" class="form-label">Weight (kg) Example: 65 </label>
          <input
            type="number"
            step="any"
            min="1"
            class="form-control"
            id="weight"
            name="weight"
            required
            value="{{ weight or '' }}"
          />
        </div>
        <div class="mb-3">
          <label for="height" class="form-label">Height (cm) Example: 170 </label>
          <input
            type="number"
            step="any"
            min="1"
            class="form-control"
            id="height"
            name="height"
            required
            value="{{ height or '' }}"
          />
        </div>
        <div class="mb-3">
          <label for="age" class="form-label">Age (years) Example: 40 </label>
          <input
            type="number"
            min="1"
            class="form-control"
            id="age"
            name="age"
            required
            value="{{ age or '' }}"
          />
        </div>
        <div class="mb-3">
          <label for="gender" class="form-label">Gender</label>
          <select class="form-select" id="gender" name="gender" aria-label="Select gender">
            <option value="male" {% if gender=='male' %}selected{% endif %}>Male</option>
            <option value="female" {% if gender=='female' %}selected{% endif %}>Female</option>
          </select>
        </div>
        {% if error %}
          <div class="alert alert-danger">{{ error }}</div>
        {% endif %}
        <button type="submit" class="btn btn-primary w-100">Check Health</button>
      </form>

      {% if result %}
      <div class="mt-4" id="result-section">
        <h3 class="text-center text-primary fw-bold" style="font-size: 1.5rem;">Your Health Status</h3>
        <p><strong>BMI:</strong> {{ result.bmi }}</p>
        <p>
          <strong>Category (Rule-based):</strong>
          <span class="{{
            'text-success' if result.category == 'Normal weight' else 'text-danger'
          }}">
            {{ result.category }}
          </span>
        </p>
        <p>
          <strong>AI Predicted Category:</strong>
          <span class="{{
            'text-success' if result.ai_category == 'normal' else 'text-danger'
          }}">
            {{ result.ai_category.capitalize() }}
          </span>
        </p>

        <canvas id="bmiChart" style="max-width: 300px; margin: auto;"></canvas>

        <h5 class="mt-3">Recommendations:</h5>
        <ul>
          {% for rec in result.recommendations %}
            <li>{{ rec }}</li>
          {% endfor %}
        </ul>
      </div>
      {% endif %}
    </div>
  </div>

  <script>
    {% if result %}
    const ctx = document.getElementById('bmiChart').getContext('2d');
    const data = {
      labels: ['Underweight', 'Normal weight', 'Overweight', 'Obese'],
      datasets: [{
        label: 'BMI Category',
        data: [0, 0, 0, 0],
        backgroundColor: [
          '#63b3ed',
          '#48bb78',
          '#f6ad55',
          '#f56565',
        ],
      }]
    };
    const categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese'];
    const index = categories.indexOf("{{ result.category }}");
    if(index >= 0){
      data.datasets[0].data[index] = 1;
    }
    const config = {
      type: 'doughnut',
      data: data,
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom' },
          tooltip: { enabled: true }
        }
      }
    };
    new Chart(ctx, config);

    // Auto-scroll to result
    window.onload = function() {
      const resultSection = document.getElementById('result-section');
      if(resultSection) {
        resultSection.scrollIntoView({ behavior: 'smooth' });
      }
    };
    {% endif %}
  </script>
  
  <footer style="position: fixed; bottom: 0; width: 100%; background: #f8f9fa; text-align: center; padding: 10px 0; font-size: 0.9rem; border-top: 1px solid #ddd; color: #6c757d;">
    Developed by Muhammed Ashml Ashraf | Hosted on 
    <a href="https://railway.com" target="_blank" rel="noopener" style="text-decoration:none; color:#6c757d;">
      Railway.com
    </a>
  </footer>
</body>
</html>
"""

# === ML Model Training ===
# Sample tiny dataset with fabricated data for demo
# Features: weight (kg), height (cm), age (years), gender (0=male,1=female)
X_train = np.array([
    [50, 160, 25, 1],  # underweight female
    [70, 175, 30, 0],  # normal male
    [90, 180, 45, 0],  # overweight male
    [110, 165, 50, 1], # obese female
])
y_train = ['underweight', 'normal', 'overweight', 'obese']

# Train decision tree model once on startup
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    bmi = weight / (height_m * height_m)
    return round(bmi, 2)

def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_recommendations(category, age, gender):
    recs = [
        "Drink plenty of water.",
        "Eat a balanced diet rich in fruits and vegetables.",
        "Avoid sugary and processed foods.",
        "Exercise regularly - at least 150 minutes per week."
    ]
    if category == "Underweight":
        recs.append("Increase calorie intake with healthy fats and proteins like nuts, dairy, and lean meats.")
    elif category in ("Overweight", "Obese"):
        recs.append("Limit high-fat and high-sugar foods.")
        recs.append("Consider consulting a healthcare provider for personalized advice.")
    if age >= 50:
        recs.append("Ensure adequate calcium and vitamin D intake to support bone health.")
    if gender == "female":
        recs.append("Include iron-rich foods like spinach, lentils, and lean red meat.")
    return recs

def predict_health_category(weight, height, age, gender):
    gender_code = 0 if gender == 'male' else 1
    prediction = model.predict([[weight, height, age, gender_code]])
    return prediction[0]

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    weight = height = age = gender = None

    if request.method == "POST":
        try:
            weight = float(request.form.get("weight", "0"))
            height = float(request.form.get("height", "0"))
            age = int(request.form.get("age", "0"))
            gender = request.form.get("gender", "male")
            if weight <= 0 or height <= 0 or age <= 0 or gender not in ("male", "female"):
                error = "Please enter valid positive values."
            else:
                bmi = calculate_bmi(weight, height)
                category = classify_bmi(bmi)
                ai_category = predict_health_category(weight, height, age, gender)
                recommendations = get_recommendations(category, age, gender)
                result = {
                    "bmi": bmi,
                    "category": category,
                    "ai_category": ai_category,
                    "recommendations": recommendations,
                }
        except Exception:
            error = "Invalid input. Please check your values."

    return render_template_string(
        HTML_TEMPLATE,
        error=error,
        result=result,
        weight=weight,
        height=height,
        age=age,
        gender=gender,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
