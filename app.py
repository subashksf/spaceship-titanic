import gradio as gr
import numpy as np
import pandas as pd
import joblib
import config
from preprocessors import label_encode

with gr.Blocks() as demo:
  HomePlanet = gr.Text(label="HomePlanet")
  CryoSleep = gr.Number(label="CryoSleep")
  Destination = gr.Text(label="Destination")
  VIP = gr.Number(label="VIP")
  RoomService = gr.Number(label="RoomService")
  FoodCourt = gr.Number(label="FoodCourt")
  ShoppingMall = gr.Number(label="ShoppingMall")
  Spa = gr.Number(label="Spa")
  VRDeck = gr.Number(label="VRDeck")
  deck = gr.Text(label="deck")
  num = gr.Text(label="num")
  side = gr.Text(label="side")
  AgeGroup = gr.Text(label="AgeGroup")

  with gr.Row():
    predict_btn = gr.Button("Predict")
  
  prediction = gr.Textbox(label="Was the passenger transported?", interactive=False)

  # load the saved model
  best_model = joblib.load("./saved_models/best_model.joblib") 

  def predict(HomePlanet, 
            CryoSleep,	
            Destination,	
            VIP,	
            RoomService,
            FoodCourt,
            ShoppingMall,	
            Spa,	
            VRDeck,	
            deck,	
            num,
            side,	
            AgeGroup):

    np_array = np.array([[HomePlanet, CryoSleep, Destination, VIP,	RoomService, FoodCourt, ShoppingMall,	Spa, VRDeck,	deck,	num,side,	AgeGroup]])
    df = pd.DataFrame(np_array, columns=['HomePlanet',	'CryoSleep',	'Destination',	'VIP',	'RoomService'	,'FoodCourt'	,'ShoppingMall',	'Spa',	'VRDeck',	'deck',	'num'	,'side',	'AgeGroup'])
    
    # label encode the input parameters
    for col in config.FEATURES_TO_ENCODE:
      df = label_encode(df, col)

    transported = best_model.predict(df)

    print(f"The prediction from gradio is: {transported}")
    
    if transported == 1:
      return f"The passenger was transported"
    else:
      return f"The passenger was not transported"
  
  predict_btn.click(predict, inputs=[HomePlanet, CryoSleep,	Destination,	VIP,	RoomService	,FoodCourt	,ShoppingMall,	Spa,	VRDeck,	deck,	num	,side,	AgeGroup], outputs = [prediction])

demo.launch(debug=True)
