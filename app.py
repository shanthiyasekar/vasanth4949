import requests


api_key = '30d4741c779ba94c470ca1f63045390a'

user_input = input("Enter city: ")

weather_data = requests.get(
    f"https://api.openweathermap.org/data/2.5/weather?q={user_input}&units=imperial&APPID={api_key}")

if weather_data.json()['cod'] == '404':
    print("No City Found")
else:
    weather = weather_data.json()['weather'][0]['main']
    temp = round(((weather_data.json()['main']['temp'] - 32)*5)/9)
    humidity = weather_data.json()['main']['humidity']
    pressure=weather_data.json()['main']['pressure']
    wind_spd=weather_data.json()['wind']['speed']
    print(f"The weather in {user_input} is: {weather}")
    print(f"The temperature in {user_input} is: {temp}ºC")
    print(f"The humidity in {user_input} is: {humidity}%")
    print(f"The windspeed in {user_input} is: {wind_spd}")
    print(f"The pressure in {user_input} is: {pressure}")
w=[temp,weather,humidity,wind_spd,pressure]

f1=open('models/FNXGBoost.pkl','rb')
f2=open('models/CRNRandomForest.pkl','rb')
f3=open('models/CYFRandomForestRegressor.pkl','rb')

model1=pickle.load(f1)
model2=pickle.load(f2)
model3=pickle.load(f3)


soil_type=['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
crop_type=['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds','Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']
output1=['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP','Urea']
data = np.array([[soil_type.index('Clayey'),crop_type.index('Pulses'),24,0,19]])
prediction1 = model1.predict(data)

output2=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

data = np.array([[63.0,	42,	21.0,	w.index(temp),	w.index(humidity),	5.798424,	67.102251	]])
prediction2=model2.predict(data)

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
data = np.array([[state_name.index('Maharashtra'),district_names.index(user_input),2004,season_names.index('Kharif     '),crop_names.index('Arhar/Tur'),12200.0,w.index(temp),w.index(wind_spd),w.index(pressure),w.index(humidity),soil_type.index('peaty'),10.500,27.300,27.300,4800.0]])
prediction3 = model3.predict(data)       
output3=prediction3

print("the fertiliser for the given crop:")
print(output1[prediction1[0]])

print('the recommended crop is:')
print(output2[prediction2[0]])

print('the yield for the given crop is')
print(output3)
