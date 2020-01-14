from flask import  Flask
from flask import render_template
from flask import  request
from pymongo import MongoClient
import pickle
import numpy as np
import pandas as pd

client=MongoClient('localhost',27017)
db=client['mypredictor']
coll=db['coll']
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def my_form():
    return render_template('predictor.html')


@app.route('/' ,methods=['POST'])
def my_form_post():
    text=request.form['name']
    text2=request.form['institutionname']
    text3=request.form['g']
    text4=request.form['famrel']
    text5=request.form['health']
    text6=request.form['absent']
    text7=request.form['edum']
    text8=request.form['mjob']
    text9=request.form['eduf']
    text10=request.form['fjob']
    text11=request.form['ttime']
    text12=request.form['stime']
    text13=request.form['iaccess']
    text14=request.form['level1']
    text15=request.form['level2']

    my_dict={"Name":text,"institution_name":text2,"Gender":text3,"G1":text14,"G2":text15,"famrel":text4,"health":text5,"absent":text6,"medu":text7,"fedu":text9,"mjob":text8,"fjob":text10,"ttime":text11,"stime":text12,"iacces":text13,"G3":"null"}
    coll.insert_one(my_dict)
    
    int_features=[int(text3),int(text7),int(text9),int(text8),int(text10),int(text11),int(text12),int(text13),int(text4),int(text5),int(text6),int(text14),int(text15)]
    final_features=np.array(np.reshape(int_features,(1,13)))
    print(final_features)
    prediction=model.predict(final_features)
    output=prediction[0]
    res=(output*5)
    print(res)
    return 'YOUR GRADE FOR THIS SEMESTER MUST BE:  {}'.format(res)
    
if __name__=='__main__':
    app.debug=False
    app.run()
