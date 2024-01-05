from flask import Flask,render_template,app,request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('data.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def home():
    name = request.form['na']
    branch = request.form['branch']
    sslc = request.form['sslc']
    pu = request.form['pu']
    cgpa = request.form['cgpa']
    arr = request.form['arr']
    intern = request.form['intern']
    a=request.form.getlist('skill')
    C=0
    Cp=0
    Java=0
    Python=0
    JavaScript=0
    PHP=0
    Data_Structure=0
    DBMS=0
    Web_dev=0
    App_dev=0
    PCB=0
    if '1' in a:
        C =1
    if '2' in a:
        Cp=1
    if '3' in a:
        Java = 1
    if '4' in a:
        Python = 1
    if '5' in a:
        JavaScript = 1
    if '6' in a:
        PHP = 1
    if '7' in a:
        Data_Structure = 1
    if '8' in a:
        DBMS = 1
    if '9' in a:
        Web_dev = 1
    if '10' in a:
        App_dev = 1
    if '11' in a:
        PCB = 1
    dp = {'Branch':[branch],'SSLC':[sslc],'PUC':[pu],'BE':[cgpa],'Arrears':[arr],'Internships':[intern],'C':[C],'C++':[Cp],'Java':[Java],'Python':[Python],'JavaScript':[JavaScript],'PHP':[PHP],'Data_Structure':[Data_Structure],'DBMS':[DBMS],'Web_dev':[Web_dev],'App_dev':[App_dev],'PCB':[PCB]}
    df = pd.DataFrame(data=dp)
    pred = model.predict(df)
    pred=int(pred)
    print(pred)
    pr = 0
    if (pred):
       pr = knn_algo(branch,sslc,pu,cgpa,arr,intern,C,Cp,Java,Python,JavaScript,PHP,Data_Structure,DBMS,Web_dev,App_dev,PCB)
       print(pr)
    return render_template('index.html',data=pred,name=name,pack=pr)

def knn_algo(branch,sslc,pu,cgpa,arr,intern,C,Cp,Java,Python,JavaScript,PHP,Data_Structure,DBMS,Web_dev,App_dev,PCB):
    d = pd.read_csv('Placement_Data.csv')
    Yp = d['Package'].map({4.5:0,4:1,0.:2,3.5:3,5:4,3:5,10:6,5.5:7,6:8,6.5:9})
    Xp = d.drop(['Package','Placed'],axis=1)
    from sklearn.model_selection import train_test_split
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xp,Yp,test_size=0.2)

    from sklearn.neighbors import KNeighborsClassifier
    model1 = KNeighborsClassifier(n_neighbors = 3)
    model1.fit(Xtrain,Ytrain)
    ypre = model1.predict([[int(branch),int(sslc),int(pu),float(cgpa),int(arr),int(intern),C,Cp,Java,Python,JavaScript,PHP,Data_Structure,DBMS,Web_dev,App_dev,PCB]])
    import numpy as np
    name = np.array(['Above 4.5','Above 4','No Predictable','Above 3.5','Above 5','Above 3','Above 10','Above 5.5','Above 6','Above 6.5'])

    return name[ypre]


if __name__ == "__main__":
    app.run(debug=True)


