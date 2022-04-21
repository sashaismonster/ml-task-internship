import flask
from flask import request, jsonify
from flask import Flask, render_template
from main import to_show

app = flask.Flask(__name__, template_folder='templates')
app.config["DEBUG"] = True

# Create some test data for our catalog in the form of a list of dictionaries.
recomendations = to_show().to_dict()

@app.route('/', methods=['GET'])
def home():\
    return render_template('index.html')

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form

        return render_template('data.html',form_data = form_data)

@app.route('/api/all', methods=['GET'])
def api_all():
    return jsonify(recomendations)

@app.route('/api/users', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for book in recomendations:
        if book['id'] == id:
            results.append(book)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

app.run()