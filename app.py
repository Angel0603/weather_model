from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('modelo.pkl')

label_map = {0: 'Lluvia', 1: 'Nieve'}  # Diccionario para mapear los números a etiquetas

@app.route('/')
def home():
    # Servir la página con el formulario
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Dew_Point_Temp_C = float(request.form.get('Dew Point Temp_C', 0))
        Temp_C = float(request.form.get('Temp_C', 0))
        Press_kPa = float(request.form.get('Press_kPa', 0))
        Rel_Hum = float(request.form.get('Rel Hum_%', 0))
        Month = float(request.form.get('Month', 0))

        data = {
            'Dew Point Temp_C': [Dew_Point_Temp_C],
            'Temp_C': [Temp_C],
            'Press_kPa': [Press_kPa],
            'Rel Hum_%': [Rel_Hum],
            'Month': [Month]
        }
        data_df = pd.DataFrame(data)

        # Realizar predicciones
        prediction = model.predict(data_df)
        
        # Convertir la predicción numérica a la etiqueta correspondiente
        result = label_map[prediction[0].item()]  # .item() convierte a Python scalar y mapea

        # Devolver la predicción como JSON
        return jsonify({'categoria': result})
    except Exception as e:
        # Devuelve un mensaje de error más útil
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
