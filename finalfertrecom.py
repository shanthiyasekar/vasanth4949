from flask import Flask,render_template,request,redirect

import pickle
import pandas as pd
import numpy as np
import requests

app=Flask(__name__)

model=pickle.load(open('models\CRRandomForest.pkl','rb'))
df=pd.read_csv('data\Crop_recommendation.csv')

model2=pickle.load(open('models\FNSupport Vector.pkl','rb'))
df1=pd.read_csv('data\ert1.csv')

model3=pickle.load(open('models\RandomForestRegressor.pkl','rb'))
df2=pd.read_csv('data\ewy.csv')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/crop')
def crop():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():  
    nitrogen=request.form.get('nitrogen')
    phosphorous=request.form.get('pho')
    pottasium=request.form.get('pot')
    ph=request.form.get('ph')
    rainfall=request.form.get('rainfall')
    user_input = request.form.get("loc ")
    api_key = '30d4741c779ba94c470ca1f63045390a'

   
    weather_data = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={user_input}&units=imperial&APPID={api_key}")

    if weather_data.json()['cod'] == '404':
        print("No City Found")
    else:
        weather = weather_data.json()['weather'][0]['main']
        temp = round(((weather_data.json()['main']['temp'] - 32)*5)/9)
        humidity = weather_data.json()['main']['humidity']
        pressure=weather_data.json()['main']['pressure']
        wind_spd=weather_data.json()['wind']['speed']
   
        w=[temp,weather,humidity,wind_spd,pressure]

    prediction=model.predict(pd.DataFrame([[nitrogen,phosphorous,pottasium,w.index(temp),w.index(humidity),ph,rainfall]],columns=['N','P','K','temperature','humidity','ph','rainfall']))
    return prediction[0]

@app.route('/fertilizer')
def fert():
    return render_template('fertilizer.html')


@app.route('/predictfert',methods=['POST'])
def predictfert():
    soil=request.form.get('soil')
    crop=request.form.get('crop')
    nitrogen=int(request.form.get('nitrogen'))
    phosphorous=int(request.form.get('pho'))
    pottasium=int(request.form.get('pot'))
    soil_type=['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
    crop_type=['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds','Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']
    output1=['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP','Urea']
    data = np.array([[soil_type.index(soil),crop_type.index(crop),nitrogen,pottasium,phosphorous]])
    prediction1 = model2.predict(data)
    print(output1[prediction1[0]])
    #print(soil,crop,nitrogen,phosphorous,pottasium)
    #prediction=model1.predict(soil,crop,nitrogen,pottasium,phosphorous)
    return output1[prediction1[0]]

@app.route('/yield')

def yiel():

    state=sorted(df2['state_names'].unique())
    district=sorted(df2['district_names'].unique())
    year=sorted(df2['crop_year'].unique(),reverse=True)
    season=sorted(df2['season_names'].unique())
    crop=sorted(df2['crop_names'].unique())
    soil=sorted(df2['soil_type'].unique())

    return render_template('yield.html',states=state,districts=district,years=year,seasons=season,crops=crop,soils=soil)


@app.route('/predictyield',methods=['POST'])
def predictyield():
    state=request.form.get('state')
    district=request.form.get('district')
    crop_year=int(request.form.get('year'))
    season=request.form.get('season')
    crop=request.form.get('crop')
    area=request.form.get('area')
    soil=request.form.get('soil')
    nitrogen=request.form.get('nitrogen')
    phosphorous=request.form.get('phos')
    pottasium=request.form.get('pot')
    production=request.form.get('prod')
    api_key = '30d4741c779ba94c470ca1f63045390a'

   

    weather_data = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={district}&units=imperial&APPID={api_key}")

    if weather_data.json()['cod'] == '404':
        print("No City Found")
    else:
        weather = weather_data.json()['weather'][0]['main']
        temp = round(((weather_data.json()['main']['temp'] - 32)*5)/9)
        humidity = weather_data.json()['main']['humidity']
        pressure=weather_data.json()['main']['pressure']
        wind_spd=weather_data.json()['wind']['speed']
   
    w=[temp,weather,humidity,wind_spd,pressure]

    state_name=['Maharashtra']
    district_names=['AHMEDNAGAR', 'AKOLA', 'AMRAVATI', 'AURANGABAD', 'BEED',
       'BHANDARA', 'BULDHANA', 'CHANDRAPUR', 'DHULE', 'GADCHIROLI',
       'GONDIA', 'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR', 'LATUR',
       'NAGPUR', 'NANDED', 'NANDURBAR', 'NASHIK', 'OSMANABAD', 'PALGHAR',
       'PARBHANI', 'PUNE', 'RAIGAD', 'RATNAGIRI', 'SANGLI', 'SATARA',
       'SINDHUDURG', 'SOLAPUR', 'THANE', 'WARDHA', 'WASHIM', 'YAVATMAL']
    season_names=['Kharif     ', 'Rabi       ', 'Summer     ', 'Whole Year ']   
    crop_names=['Arhar/Tur', 'Bajra', 'Groundnut', 'Jowar', 'Maize',
       'Moong(Green Gram)', 'Niger seed', 'Other Cereals & Millets',
       'Other Kharif pulses', 'other oilseeds', 'Ragi', 'Rice', 'Sesamum',
       'Sunflower', 'Urad', 'Gram', 'Linseed', 'Other  Rabi pulses',
       'Rapeseed &Mustard', 'Safflower', 'Soyabean', 'Wheat',
       'Cotton(lint)', 'Sugarcane', 'Castor seed', 'Tobacco']  
    soil_type=['loamy', 'sandy', 'clay', 'chalky', 'peaty', 'silty', 'silt']       
  
    #data = np.array([[state_name.index(state),district_names.index(district),crop_year,season_names.index(season),crop_names.index(crop),area,w.index(temp),w.index(wind_spd),w.index(pressure),w.index(humidity),soil_type.index(soil),nitrogen,phosphorous,pottasium,production]])
  
    #data = np.array([[state_name.index('Maharashtra'),district_names.index(district),2004,season_names.index('Kharif     '),crop_names.index('Arhar/Tur'),12200.0,w.index(temp),w.index(wind_spd),w.index(pressure),w.index(humidity),soil_type.index('peaty'),10.500,27.300,27.300,4800.0]])
    #prediction3 = model3.predict(data)  
    #print(prediction3)     
    #return str(prediction3[0])

    prediction=model3.predict(pd.DataFrame([[state_name.index(state),district_names.index(district),crop_year,season_names.index(season),crop_names.index(crop),area,w.index(temp),w.index(wind_spd),w.index(pressure),w.index(humidity),soil_type.index(soil),nitrogen,phosphorous,pottasium,production]],columns=['state_names','district_names','crop_year','season_names','crop_names','area','temperature','wind_speed','pressure','humidity','soil_type','N','P','K','production']))
                         
    print(prediction)
    return str(prediction[0])


if __name__=='__main__':
    app.run(debug=True)    
