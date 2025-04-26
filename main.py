from flask import Flask, render_template, request, flash
import pickle
import json
import numpy as np
import CONFIG

with open(CONFIG.MODEL_PATH,'rb') as file:
    model = pickle.load(file)

with open(CONFIG.ASSET_PATH,'r') as file:
    asset = json.load(file)
col = asset['columns']
app = Flask(__name__)
app.secret_key = 'iris_classification_key'  # Required for flash messages

# Mapping of class indices to species names and descriptions
IRIS_SPECIES = {
    0: {
        "name": "SETOSA",
        "description": "Characterized by small flowers with wide sepals and narrow petals."
    },
    1: {
        "name": "VERSICOLOR",
        "description": "Medium-sized flowers with moderate sepal and petal dimensions."
    },
    2: {
        "name": "VIRGINICA",
        "description": "Distinguished by larger flowers with longer petals."
    }
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get_data", methods = ["POST"])
def data():
    try:
        input_data = request.form
        
        # Validate input data
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                flash(f"Error: {field.replace('_', ' ').title()} is required")
                return render_template("index.html")
            
        # Convert and validate numerical values
        measurements = {}
        for field in required_fields:
            try:
                value = float(input_data[field])
                if value <= 0 or value > 10:
                    flash(f"Error: {field.replace('_', ' ').title()} must be between 0 and 10 cm")
                    return render_template("index.html")
                measurements[field] = value
            except ValueError:
                flash(f"Error: {field.replace('_', ' ').title()} must be a valid number")
                return render_template("index.html")
        
        # Prepare data for prediction
        data = np.zeros(len(col))
        data[0] = measurements['sepal_length']
        data[1] = measurements['sepal_width']
        data[2] = measurements['petal_length']
        data[3] = measurements['petal_width']
        
        # Make prediction
        result = model.predict([data])
        prediction_index = result[0]
        
        # Get species information
        species_info = IRIS_SPECIES.get(prediction_index, {"name": "UNKNOWN", "description": ""})
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([data])[0]
            probabilities = {IRIS_SPECIES[i]["name"]: f"{p:.2%}" for i, p in enumerate(proba)}
        
        return render_template(
            "index.html",
            PREDICT_VALUE=species_info["name"],
            DESCRIPTION=species_info["description"],
            PROBABILITIES=probabilities,
            INPUT_DATA=measurements
        )
        
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host=CONFIG.HOST_NAME, port=CONFIG.PORT_NUMBER, debug=True)