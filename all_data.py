#Important Modules
from flask import Flask,render_template, url_for ,flash , redirect
#from forms import RegistrationForm, LoginForm
#from sklearn.externals import joblib
import joblib
from joblib import dump, load
from flask import request
import numpy as np
import tensorflow

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')



# RELATED TO THE SQL DATABASE
#app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
#db=SQLAlchemy(app)

#from model import User,Post

#//////////////////////////////////////////////////////////

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

#graph = tf.get_default_graph()
#with graph.as_default():;
from tensorflow.keras.models import load_model
model = load_model('model111.h5')
#model222=load_model("my_model.h5")
model222=load_model("model_vgg16.h5")

#FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted
#FOR THE SECOND MODEL
def api1(full_path):
   # data = image.load_img(full_path, target_size=(64, 64, 3))
    data = image.load_img(full_path, target_size=(224, 224, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model222.predict(data)
    return predicted


# home page

#@app.route('/')
#def home():
 #  return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))

@app.route('/upload11', methods=['POST','GET'])
def upload11_file():

    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            
            if(result>50):
                label= indices[1]
                accuracy= result
                
            else:
                label= indices[0]
                accuracy= 100-result
            print(result)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]

            return render_template('predict1.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


#@app.route("/")

#@app.route("/home")
#def home():
#    return render_template("home.html")

#@app.route("/about")
#def about():
#   return render_template("about.html")


#@app.route("/cancer")
#def cancer():
    #return render_template("cancer.html")


#@app.route("/diabetes")
#def diabetes():
    #if form.validate_on_submit():
    #return render_template("diabetes.html")

#@app.route("/heart")
#def heart():
    #return render_template("heart.html")


#@app.route("/liver")
#def liver():
    #if form.validate_on_submit():
    #return render_template("liver.html")

#@app.route("/kidney")
#def kidney():
    #if form.validate_on_submit():
    #return render_template("kidney.html")

#@app.route("/Malaria")
#def Malaria():
    #return render_template("index.html")

#@app.route("/Pneumonia")
#def Pneumonia():
    #return render_template("index2.html")



"""def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size==12):#Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size==10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("model2")
        result =loaded_model.predict(to_predict)
    return result[0]"""

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = int(input('input_val'))
        #Cancer
        if(to_predict_list==1):
        #if(len(to_predict_list)==30):#Cancer
            #result = ValuePredictor(to_predict_list,30)
             #cancer
             radius_mean = int(input('radius_mean'))  
             texture_mean = int(input('texture_mean'))  
             perimeter_mean = int(input('perimeter_mean'))  
             area_mean = int(input('area_mean'))  
             smoothness_mean = int(input('smoothness_mean'))  
             compactness_mean = int(input('compactness_mean'))  
             concavity_mean = int(input('concavity_mean'))  
             concave_points_mean = int(input('concave_points_mean'))  
             symmetry_mean = int(input('symmetry_mean'))  
             fractal_dimension_mean = int(input('fractal_dimension_mean'))  
             radius_se = int(input('radius_se'))  
             texture_se = int(input('texture_se'))  
             perimeter_se = int(input('perimeter_se'))  
             area_se = int(input('area_se'))  
             smoothness_se = int(input('smoothness_se'))  
             compactness_se = int(input('compactness_se'))  
             concave_points_se = int(input('concave_points_se'))  
             symmetry_se = int(input('symmetry_se'))  
             fractal_dimension_se = int(input('fractal_dimension_se'))  
             radius_worst = int(input('radius_worst'))  
             texture_worst = int(input('texture_worst'))  
             perimeter_worst = int(input('perimeter_worst')) 
             area_worst = int(input('area_worst')) 
             smoothness_worst = int(input('smoothness_worst')) 
             compactness_worst = int(input('compactness_worst')) 
             concavity_worst = int(input('concavity_worst')) 
             concave_points_worst = int(input('concave_points_worst')) 
             symmetry_worst = int(input('symmetry_worst')) 
             fractal_dimension_worst = int(input('fractal_dimension_worst')) 
             lst = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,
                    concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,texture_se,perimeter_se,area_se,
                    smoothness_se,compactness_se,concave_points_se,symmetry_se,fractal_dimension_se,
                    radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,
                    concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst,]
             
             to_predict_list = list(map(float, lst))
             to_predict = np.array(to_predict_list).reshape(1,30)
             loaded_model = joblib.load("model")
             result = loaded_model.predict(to_predict)
             
        #Diabetis    
        elif(to_predict_list==2):
            #result = ValuePredictor(to_predict_list,8)
            pregnancies = int(input('pregnancies'))
            glucose = int(input('glucose'))
            bloodpressure = int(input('bloodpressure'))
            skinthickness = int(input('skinthickness'))
            insulin = int(input('insulin'))
            bmi = int(input('bmi'))
            diabetespedigreefunction = int(input('diabetespedigreefunction'))
            age = int(input('age'))
            
            lst = [pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]
            to_predict_list = list(map(float, lst))
            to_predict = np.array(to_predict_list).reshape(1,8)
            loaded_model = joblib.load("model1")
            result = loaded_model.predict(to_predict)
            
        #kidney
        elif(to_predict_list==3):
            #result = ValuePredictor(to_predict_list,12)

            age = int(input('age')) 
            bp = int(input('bp')) 
            al = int(input('al'))
            pcc = int(input('pcc'))
            bgr = int(input('bgr'))
            bu = int(input('bu'))
            sc = int(input('sc'))
            hemo = int(input('hemo'))
            pcv = int(input('pcv'))
            htn = int(input('htn'))
            dm = int(input('dm'))
            appet = int(input('appet'))
            
            lst = [age,bp,al,pcc,bgr,bu,sc,hemo,pcv,htn,dm,appet]
            to_predict_list = list(map(float, lst))
            to_predict = np.array(to_predict_list).reshape(1,12)
            loaded_model = joblib.load("model3")
            result = loaded_model.predict(to_predict)
            
        #heart    
        elif(to_predict_list==4):
            #result = ValuePredictor(to_predict_list,11)
            
            age = int(input('age')) 
            sex = int(input('sex')) 
            chest_pain_type = int(input('chest_pain_type')) 
            trestbps = int(input('trestbps')) 
            serum_cholestoral = int(input('serum_cholestoral')) 
            restecg = int(input('restecg')) 
            thalach = int(input('thalach')) 
            exang = int(input('exang')) 
            oldpeak = int(input('oldpeak')) 
            slope = int(input('slope')) 
            thal = int(input('thal')) 
            
            lst = [age,sex,chest_pain_type,trestbps,serum_cholestoral,restecg,thalach,exang,oldpeak,slope,thal]
            to_predict_list = list(map(float, lst))
            to_predict = np.array(to_predict_list).reshape(1,11)
            loaded_model = joblib.load("model2")
            result = loaded_model.predict(to_predict)
            
        #liver     
        elif(to_predict_list==5):
            #result = ValuePredictor(to_predict_list,10)
            age = int(input('age'))
            gender = int(input('gender'))
            total_bilirubin = int(input('total_bilirubin'))
            direct_bilirubin = int(input('direct_bilirubin'))
            alkaline_phosphotase = int(input('alkaline_phosphotase'))
            alamine_aminotransferase = int(input('alamine_aminotransferase'))
            aspartate_aminotransferase = int(input('aspartate_aminotransferase'))
            total_protiens = int(input('total_protiens'))
            albumin = int(input('albumin'))
            albumin_and_globulin_ratio = int(input('albumin_and_globulin_ratio'))
            
            lst = [age,gender,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_protiens,albumin,albumin_and_globulin_ratio]
            to_predict_list = list(map(float, lst))
            to_predict = np.array(to_predict_list).reshape(1,10)
            loaded_model = joblib.load("model4")
            result = loaded_model.predict(to_predict)
            
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)


#set FLASK_ENV = development
#flask run