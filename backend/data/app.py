from flask import Flask, render_template
import csv

app = Flask(__name__)

@app.route('/')
def show_data():
    data = []
    with open('./co2_data.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) == 2:
                data.append({'timestamp': row[0], 'co2': float(row[1])})
    return render_template('index.html', data=data)

@app.route('/table')
def table():
    data = []
    with open('./co2_data.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) == 2:
                data.append({'timestamp': row[0], 'co2': float(row[1])})
    return render_template('table.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
