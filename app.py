from flask import Flask, request, render_template, jsonify
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Superclass labels
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Define recommendations
first_recommendations = {
    "MI": "Consider possible myocardial infarction.",
    "STTC": "Light ST/T changes detected. Consider consulting a cardiologist.",
    "CD": "Possible conduction disturbance. Further evaluation recommended.",
    "HYP": "Possible hypertrophy. Regular monitoring advised."
}

second_recommendations = {
    "MI": "Seek immediate medical attention for myocardial infarction",
    "STTC": "ST/T changes detected. Consult a cardiologist.",
    "CD": "Conduction disturbance detected.",
    "HYP": "Hypertrophy detected. Regular monitoring advised."
}

@app.route('/')
def index():
    return render_template('index.html')  # Adjusted path for templates

@app.route('/upload', methods=['POST'])
def upload():
    if 'ecg_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['ecg_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Simulate loading and preprocessing data (replace with actual preprocessing)
        # Here, we'll generate fake probabilities that sum up to 1 (100%)
        random_probs = np.random.rand(len(SUPERCLASSES))  # Fake probabilities (not summing to 1)
        
        # Force at least one likelihood above 60% (let's set the NORM class to a high value)
        random_probs[2] = max(random_probs[2], 0.6)  # Ensure NORM is at least 60%

        # Re-normalize the probabilities so they sum up to 1
        normalized_probs = random_probs / random_probs.sum()  # Normalize so they sum to 1
        
        # Assign normalized probabilities to the respective superclasses
        likelihoods = dict(zip(SUPERCLASSES, normalized_probs))

        # Build response based on likelihood ranges
        response = []
        for class_name, likelihood in likelihoods.items():
            if class_name == "NORM":
                if likelihood > 0.6:
                    response.append({
                        "class": class_name,
                        "likelihood": f"{(likelihood * 100):.2f}%",
                        "recommendation": ""
                    })
                # Skip adding the NORM row if likelihood <= 0.6
                continue

            if likelihood < 0.15:
                response.append({
                    "class": class_name,
                    "likelihood": f"{(likelihood * 100):.2f}%",
                    "recommendation": "No abnormalities detected."
                })
            elif 0.15 <= likelihood < 0.6:
                response.append({
                    "class": class_name,
                    "likelihood": f"{(likelihood * 100):.2f}%",
                    "recommendation": first_recommendations[class_name]
                })
            else:
                response.append({
                    "class": class_name,
                    "likelihood": f"{(likelihood * 100):.2f}%",
                    "recommendation": second_recommendations[class_name]
                })

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
