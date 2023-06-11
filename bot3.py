import telebot
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

# Create a new bot object
bot = telebot.TeleBot('6251632293:AAEtVn_bjV9i7rimZfcX4sAE8t-GmFD4c6Y')

dataset = pd.read_csv("F:/MyBot/HateSpeechData.csv")

# Create a CountVectorizer object and fit it on the dataset
cv = CountVectorizer()
cv.fit_transform(dataset["tweet"])
# Load the pre-trained hate speech and offensive language detection model
with open('F:/MyBot/EnglishModelHateSpeechModelMain.pkl', 'rb') as f:
    model = pickle.load(f)
# Define a message handler function
@bot.message_handler(func=lambda message: True ,content_types=['text'])
def handle_message(message):
    # if message.chat.type in ['channel', 'supergroup']:
       dp = cv.transform([message.text]).toarray()
       dt = model.predict(dp)[0]
       if dt =='Hate Speech':
           bot.delete_message(message.chat.id, message.message_id)
           bot.send_message(message.chat.id,"///////////////////////////////") 
           bot.send_message(message.chat.id,"/////the post is blocked//////")
           bot.send_message(message.chat.id,"//////////////////////////////")
   
       elif dt =='Offensive Speech':
           bot.send_message(message.chat.id,"Warnnig.....")
           bot.send_message(message.chat.id,"Warnnig.....")
           bot.send_message(message.chat.id,"Warnnig.....")
    
       
# Start the bot
bot.polling()
