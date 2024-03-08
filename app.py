import gradio as gr
import numpy as np
import pandas as pd

with gr.Blocks() as demo:
  HomePlanet = gr.Number(label="HomePlanet")
  CryoSleep = gr.Number(label="CryoSleep")
  Destination = gr.Number(label="Destination")
  VIP = gr.Number(label="VIP")
  RoomService = gr.Number(label="RoomService")
  FoodCourt = gr.Number(label="FoodCourt")
  ShoppingMall = gr.Number(label="ShoppingMall")
  Spa = gr.Number(label="Spa")
  VRDeck = gr.Number(label="VRDeck")
  deck = gr.Number(label="deck")
  num = gr.Number(label="num")
  side = gr.Number(label="side")
  AgeGroup = gr.Number(label="AgeGroup")

  with gr.Row():
    predict_btn = gr.Button("Predict")
  
  prediction = gr.Textbox(label="Was the passenger transported?", interactive=False)

  def predict(HomePlanet=0, 
            CryoSleep=0,	
            Destination=0,	
            VIP=0,	
            RoomService=0,
            FoodCourt=0,
            ShoppingMall=0,	
            Spa=0,	
            VRDeck=0,	
            deck=0,	
            num=0,
            side=0,	
            AgeGroup=0):
    np_array = create_array([[num1, num2, num3, num4, num5, num6, num7, num8, num9, num10, num11, num12, num13]])
    df = pd.DataFrame(np_array, columns=['HomePlanet',	'CryoSleep',	'Destination',	'VIP',	'RoomService'	,'FoodCourt'	,'ShoppingMall',	'Spa',	'VRDeck',	'deck',	'num'	,'side',	'AgeGroup'])
    transported = rfc.predict(df)[0]
    if transported == 1:
      return f"The passenger was transported"
    else:
      return f"The passenger was not transported"

  def create_array(numbers):
    # Convert the list of numbers to a NumPy array
    array = np.array(numbers)
    return array
  
  predict_btn.click(predict, inputs=[HomePlanet, CryoSleep,	Destination,	VIP,	RoomService	,FoodCourt	,ShoppingMall,	Spa,	VRDeck,	deck,	num	,side,	AgeGroup], outputs = [prediction])

demo.launch(debug=True)
