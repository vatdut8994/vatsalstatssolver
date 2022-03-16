from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from io import BytesIO
import statistics as stats
import base64


plt.switch_backend('Agg') 

app = Flask(__name__)


@app.route('/')
def start():
    return render_template('stats.html')

@app.route('/answers', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        global pyplot
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
        try:
            centraltendency = str(request.form['card'])
            list1 = centraltendency.split(',')
            for i in range(len(list1)):
                list1[i] = float(list1[i])
            centraltendency = list1
            statans=f"Mean: {stats.mean(centraltendency)},\nMedian: {stats.median(centraltendency)},\nMode: {stats.mode(centraltendency)},\nStandard Deviation: {stats.stdev(centraltendency)}\n"
        except:
            statans= ''
        if graphing == 1:
            pyplot = plt.plot(x, y)
        elif graphing == 0:
            pyplot = plt.scatter(x, y)
        line_of_best_fit = plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.savefig('./static/images/new_plot.png')
        data = pd.DataFrame({"x":x, "y":y })
        list1 = line_of_best_fit[0].get_data()
        print(x)
        x = list1[0]
        y = list1[1]
        print(y)
        line_table = pd.DataFrame({'x':x, 'y':y})

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
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        return render_template('stats.html', data_table=data, corr=data.corr(), graph='/static/images/new_plot.png', answer=model.predict([test]), plot_url=plot_url, bestfit=line_table, centraltendencies=statans)

if __name__ == '__main__':
    app.run(debug=True)
