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
  
  prediction = gr.Number(label="Prediction")

  def predict(num1, num2, num3, num4, num5, num6, num7, num8, num9, num10, num11, num12, num13):
    np_array = create_array([[num1, num2, num3, num4, num5, num6, num7, num8, num9, num10, num11, num12, num13]])
    df = pd.DataFrame(np_array, columns=['HomePlanet',	'CryoSleep',	'Destination',	'VIP',	'RoomService'	,'FoodCourt'	,'ShoppingMall',	'Spa',	'VRDeck',	'deck',	'num'	,'side',	'AgeGroup'])
    prediction = rand.predict(df)
    return prediction

  def create_array(numbers):
    # Convert the list of numbers to a NumPy array
    array = np.array(numbers)
    return array
  
  predict_btn.click(predict, inputs=[HomePlanet, CryoSleep,	Destination,	VIP,	RoomService	,FoodCourt	,ShoppingMall,	Spa,	VRDeck,	deck,	num	,side,	AgeGroup], outputs=prediction)

demo.launch(debug=True)
