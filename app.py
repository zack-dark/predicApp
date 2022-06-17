import email
from flask import Flask, render_template, request,session,url_for,flash
#from flask_mail import Mail,Message
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import false
from config import mail_username
import smtplib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix





def pre_traitement(database, colonne, val_qualitative, val_quantitative):
	db = database[colonne].replace(val_qualitative, val_quantitative, inplace=True)
	return db


def creer_modele(n_nei, m):
	model = KNeighborsClassifier(n_neighbors=n_nei, metric=m)
	return model


def entrainer_modele(modele, x, y):
	modele.fit(x, y.ravel())


def score_modele(modele, x, y):
	R = modele.score(x, y)
	return R


def cross_validation(x, y):
	param_grid = {'n_neighbors': np.arange(1, 30), 'metric': ['euclidean', 'manhattan', 'minkowski']}
	grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
	grid.fit(x, y.ravel())
	M_estimateur = grid.best_estimator_
	return M_estimateur


def predire(modele, param):
	x = np.array(param).reshape(1, 11)
	pred_val = modele.predict(x)
	proba = modele.predict_proba(x)
	p = round(proba[0][1] * 100, 2)
	pn = round(proba[0][0] * 100, 2)
	if pred_val == 1:
		#print('Cette personne a une maladie cardiaque avec une probabilité de : ', p, '%')
		p=['Cette personne a une maladie cardiaque avec une probabilité de: ',str(p)]
		return p
	else:
		#print('Cette personne est normale avec une probabilité de : ', pn, '%')
		p=['Cette personne est normale avec une probabilité de: ',str(pn)]
		return p



def main(param):
	"""
	Importation de la base de données
	"""

	data = pd.read_csv('database.csv')
	data = data.dropna(axis=0)

	"""
	Prétraitement des données :
	Remplacer les valeurs des parametres qualitatives par des valeurs quantitatives
	"""

	pre_traitement(data, 'Sex', ['M', 'F'], [0, 1])
	# M : Masculin 0 / F : Féminin 1
	pre_traitement(data, 'ChestPainType', ['TA', 'ATA', 'NAP', 'ASY'], [0, 1, 2, 3])
	# TA : Angine Typique 0 / ATA : Angine ATypique 1
	# NAP : Douleur Non Angineuse 2 / ASY : ASYmpomatique 3
	pre_traitement(data, 'RestingECG', ['Normal', 'ST', 'HVG'], [0, 1, 2])
	# Normal : normal 0 / ST : anomalie de l'onde ST-T 1
	# HVG : Hypertrophie Ventriculaire Gauche 2
	pre_traitement(data, 'ExerciseAngina', ['N', 'Y'], [0, 1])
	# N : Non 0 / Y : Oui 1
	pre_traitement(data, 'ST_Slope', ['Up', 'Flat', 'Down'], [0, 1, 2])
	# Up : ascendant 0 / Flat : plat 1 / Down : descendant 2

	"""
	Création des paramètres et des valeurs cibles
	"""

	parameters = data[
		['Age', 'ExerciseAngina', 'ST_Slope', 'Sex', 'ChestPainType', 'RestingECG', 'Oldpeak', 'RestingBP',
		 'Cholesterol', 'MaxHR', 'FastingBS']]
	predict_value = data['HeartDisease']

	"""
	Diviser la base de données en données d'entrainement et en données de test
	"""

	parameters_train, parameters_test, predict_value_train, predict_value_test = train_test_split(parameters,
																								  predict_value,
																								  test_size=0.2,
																								  random_state=5)

	"""
	Création et entrainement du modèle
	"""

	modele1 = creer_modele(1, 'euclidean')
	entrainer_modele(modele1, parameters_train, predict_value_train)
	R1 = score_modele(modele1, parameters_test, predict_value_test)
	########print('Le coefficient de detarmination du modele avant optimisation est : ', round(R1, 3))

	"""
	Optimisation du modèle en cherchant les valeurs optimales des paramètres
	n_neighbors,metric de la fonction KNeighborsClassifier()
	"""
	modele2 = cross_validation(parameters_train, predict_value_train)
	R2 = score_modele(modele2, parameters_test, predict_value_test)
	##########print('Le coefficient de detarmination du modele apres optimisation est : ', round(R2, 3))

	"""
	Evaluation de la qualité de classification
	"""

	M_confusion = confusion_matrix(predict_value_test, modele2.predict(parameters_test))
	######print("la matrice de confusion : ")
	######print(M_confusion)

	"""
	Faire des prédictions
	"""

	predire(modele2, param)


# if __name__ == '__main__':
# 	main([60, 1, 1, 0, 3, 0, 0.5, 120, 150, 170, 1])

"""
Exemples


~~ Atteints d'une maladie

main([76,0,2,1,1,2,1.5,180,300,120,0])
main([21,1,1,0,3,0,0,120,100,120,0])
main([30,1,1,0,3,0,1.5,200,60,80,0])
main([30,1,1,0,3,0,1.5,200,60,180,0])

~~ En etat normale

main([30,1,1,0,3,0,1,100,170,200,1])
main([60,1,1,0,3,0,0.5,120,150,170,1])
main([60,1,1,0,3,0,0.5,120,150,140,1])
"""

app=Flask(__name__)
app.config['SECRET_KEY']="HELLo"
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///users.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=false
# app.config['MAIL_SERVER']="smtp.gmail.com"
# app.config['MAIL_PORT']=465
# app.config['MAIL_USE_TLS']=False
# app.config['MAIL_USE_SSL']=True
# app.config['MAIL_USERNAME']=mail_username
# app.config['MAIL_PASSWORD']=mail_password

#mail=Mail(app)
db=SQLAlchemy(app)

class users(db.Model):
	_id=db.Column("id",db.Integer,primary_key=True)
	name=db.Column(db.String(100))
	email=db.Column(db.String(100))
	phone=db.Column(db.String(15))
	message=db.Column(db.String(250))

	def __init__(self,name,email,phone,message):
		self.name=name
		self.email=email
		self.phone=phone
		self.message=message






@app.route('/')
def home():
	return render_template('index.html')

@app.route('/form')
def form():
	return render_template('form.html')

@app.route('/contact',methods=['GET','POST'])
def contact():
	if request.method=="POST":
		#session.permanent=True

		#name=request.form['name']
		email=request.form['email']
		phone=request.form['phone']
		message=request.form['message']
		user=request.form["name"]
		session['user']=user

		found_user=users.query.filter_by(name=user).first()

		if found_user:
			session["email"]=found_user.email
		else:
			usr=users(user,email,phone,message)
			db.session.add(usr)
			db.session.commit()

		server=smtplib.SMTP("smtp.gmail.com",587)
		server.starttls()
		server.login("penitration.testing2022@gmail.com","password")
		server.sendmail("penitration.testing2022@gmail.com",email,message)
		# msg=Message(subject=f"Mail from {name}",body=f"Name: {name}\nE-Mail: {email}\nPhone:{phone}\nMessage: {message}",sender='noreply@demo.com',recipients=['penitration.testing2022@gmail.com'])
		# mail.send(msg)
		return "send email"
		


	return render_template('contact.html')

@app.route('/resultat',methods=['POST','GET'])
def result():
	"""
	Importation de la base de données
	"""

	data = pd.read_csv('database.csv')
	data = data.dropna(axis=0)

	"""
	Prétraitement des données :
	Remplacer les valeurs des parametres qualitatives par des valeurs quantitatives
	"""

	pre_traitement(data, 'Sex', ['M', 'F'], [0, 1])
	# M : Masculin 0 / F : Féminin 1
	pre_traitement(data, 'ChestPainType', ['TA', 'ATA', 'NAP', 'ASY'], [0, 1, 2, 3])
	# TA : Angine Typique 0 / ATA : Angine ATypique 1
	# NAP : Douleur Non Angineuse 2 / ASY : ASYmpomatique 3
	pre_traitement(data, 'RestingECG', ['Normal', 'ST', 'HVG'], [0, 1, 2])
	# Normal : normal 0 / ST : anomalie de l'onde ST-T 1
	# HVG : Hypertrophie Ventriculaire Gauche 2
	pre_traitement(data, 'ExerciseAngina', ['N', 'Y'], [0, 1])
	# N : Non 0 / Y : Oui 1
	pre_traitement(data, 'ST_Slope', ['Up', 'Flat', 'Down'], [0, 1, 2])
	# Up : ascendant 0 / Flat : plat 1 / Down : descendant 2

	"""
	Création des paramètres et des valeurs cibles
	"""

	parameters = data[
		['Age', 'ExerciseAngina', 'ST_Slope', 'Sex', 'ChestPainType', 'RestingECG', 'Oldpeak', 'RestingBP',
		 'Cholesterol', 'MaxHR', 'FastingBS']]
	predict_value = data['HeartDisease']

	"""
	Diviser la base de données en données d'entrainement et en données de test
	"""

	parameters_train, parameters_test, predict_value_train, predict_value_test = train_test_split(parameters,
																								  predict_value,
																								  test_size=0.2,
																								  random_state=5)

	"""
	Création et entrainement du modèle
	"""

	modele1 = creer_modele(1, 'euclidean')
	entrainer_modele(modele1, parameters_train, predict_value_train)
	R1 = score_modele(modele1, parameters_test, predict_value_test)
	########print('Le coefficient de detarmination du modele avant optimisation est : ', round(R1, 3))

	r1="Le coefficient de determination du modele avant optimisation est : "+ str(round(R1,3))


	"""
	Optimisation du modèle en cherchant les valeurs optimales des paramètres
	n_neighbors,metric de la fonction KNeighborsClassifier()
	"""
	modele2 = cross_validation(parameters_train, predict_value_train)
	R2 = score_modele(modele2, parameters_test, predict_value_test)
	##########print('Le coefficient de detarmination du modele apres optimisation est : ', round(R2, 3))
	r2="Le coefficient de detarmination du modele apres optimisation est : "+str(round(R2,3))

	"""
	Evaluation de la qualité de classification
	"""

	M_confusion = confusion_matrix(predict_value_test, modele2.predict(parameters_test))
	######print("la matrice de confusion : ")
	######print(M_confusion)
	r3="la matrice de confusion : "+str(M_confusion)

	"""
	Faire des prédictions
	"""









	if request.method=='POST':
			age=int(request.form['age'])
			sex=request.form['sex']
			chest=request.form['chest']
			restingecg=request.form['restingecg']
			exercise= request.form['exercise']
			stslope= request.form['stslope']
			restingbp= int(request.form['restingbp'])
			maxhr= int(request.form['maxhr'])
			cholesterol= int(request.form['cholesterol'])
			oldpeak= float(request.form['oldpeak'])
			fastingbs= int(request.form['fastingbs'])

	if sex=="HOMME":
		sex=0
	else:
		sex=1


	if chest=="TA":
		chest=0
	elif chest=="ATA":
		chest=1
	elif chest =="NAP":
		chest=2
	else:
		chest=3

	if restingecg=="Normal":
		restingecg=0
	elif restingecg=="ST":
		restingecg=1
	else:
		restingecg=2


	if exercise=="N":
		exercise=0
	else:
		exercise=1

	if stslope=="Up":
		stslope=0
	elif stslope=="Flat":
		stslope=1
	else:
		stslope=2

	parame=[age,exercise,stslope,sex,chest,restingecg,oldpeak,restingbp,cholesterol,maxhr,fastingbs]
	resuldt=predire(modele2, parame)

	return render_template('result.html',tit=resuldt[0],por=float(resuldt[1]))


@app.route('/view')
def view():
	return render_template('view.html',values=users.query.all())


if __name__=='__main__':
	db.create_all()
	# app.debug=True
	app.run(host='0.0.0.0',port=5000,debug=True)
