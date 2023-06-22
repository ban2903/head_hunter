from flask import Flask, render_template, request, send_file
from salary_prediction import SalaryPrediciton
# from config import FEATURES

app = Flask(__name__)
model = SalaryPrediciton()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        data = model.get_features(url)
        score = model.predict(data)[0][0]
        resources = {'meta': []}
        for c in ['real_salary_from', 'real_salary_to']+model.FEATURES:
            resources['meta'].append(
                {
                    'label': c,
                    'value': data[c].loc[0],
                }
            )
        model.set_dataset(data)
        return render_template('result.html', resources=resources, score=score)
    
    return render_template('form.html')

temp_file_path = 'scratch.pdf'

@app.route('/generate', methods=['GET'])
def generate_pdf():
    model.shap_plot(model.dataset)
    return 'done'

@app.route('/pdf-result', methods=['GET'])
def send_pdf():
    # Send the file to the user's browser
    return send_file(temp_file_path, attachment_filename='shap_plot.pdf')

if __name__ == '__main__':
    app.run()