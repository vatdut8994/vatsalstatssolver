from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
import pandas as pd

plt.switch_backend('Agg') 

app = Flask(__name__)


@app.route('/')
def start():
    return render_template('stats.html')

@app.route('/answers', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        y = str(request.form['traindata'])
        list1 = y.split(',')
        for i in range(len(list1)):
            list1[i] = float(list1[i])
        y = list1
        x = str(request.form['testdata'])
        list1 = x.split(',')
        for i in range(len(list1)):
            list1[i] = float(list1[i])
        x = list1
        graphing = int(request.form['graphing'])
        if graphing == '1':
            pygraph = plt.plot(x, y)
        else:
            pygraph = plt.scatter(x, y)
        plt.savefig('./static/images/new_plot.png')
        data = pd.DataFrame({"x":x, "y":y })
        print("Data: \n", data)

        print("Correlation Value: \n", data.corr())

        x = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

        model = LinearRegression()
        model.fit(xtrain, ytrain)

        test = str(request.form['answer'])
        list1 = test.split(',')
        for i in range(len(list1)):
            list1[i] = float(list1[i])
        test = list1
        print("Answer: ", model.predict([test]))
        return render_template('stats.html', data_table=data, corr=data.corr(), answer=model.predict([test]))

if __name__ == '__main__':
    app.run(debug=True)
