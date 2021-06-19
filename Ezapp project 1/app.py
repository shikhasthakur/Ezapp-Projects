from flask import Flask, render_template, request, redirect, session , url_for, jsonify
import mysql.connector
import numpy as np
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
app.secret_key=os.urandom(24)

conn = mysql.connector.connect(host="localhost", user="root", password="", database="flask")
cursor = conn.cursor()

@app.route('/')
def login(): 
    return render_template("login.html")

@app.route('/register')
def about(): 
    return render_template("register.html")

@app.route('/home')
def home(): 
    if 'user_id' in session:
        return render_template("index.html")
    else:
        return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email=request.form.get('email')
    password=request.form.get('password')

    cursor.execute("""SELECT * FROM `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}' """
                   .format(email, password))
    users=cursor.fetchall()
    if len(users)>0:
        session['user_id']=users[0][0]
        return redirect('/home')
    else:
        return redirect('/')

@app.route('/add_user', methods=['POST'])
def add_user():
    name=request.form.get('uname')  
    email=request.form.get('uemail') 
    password=request.form.get('upassword') 

    cursor.execute("""INSERT INTO `users` (`user_id`, `name`, `email`, `password`) VALUES(NULL, '{}','{}','{}')""".format(name,email,password))
    conn.commit()
    cursor.execute("""SELECT * FROM `users` WHERE `email` LIKE '{}'""".format(email))
    myuser=cursor.fetchall()
    session['user_id']=myuser[0][0]
    return redirect('/home')
    #return users

    #return "The email is {} and the password is {}".format(email, password)

@app.route('/Logout')
def logout():
    session.pop('user_id')
    return redirect('/')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
