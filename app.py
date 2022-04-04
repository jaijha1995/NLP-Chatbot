from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import string,re
import nltk
import os

#Flask is a web Framework
#================================================
import pickle
import numpy as np
from tensorflow import keras

# importing "windows classifier model" model
model = keras.models.load_model('win_chat_model')

# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)

# parameters
max_len = 80

# type of OS
wins = ["window10","window7"]
#================================================
nltk.download('stopwords')
app = Flask(__name__)


swords = stopwords.words('english')

# finding type of windows name
winName = [i[:-5] for i in os.listdir("data")]
winName.remove("c")
winProblems = {}


#======= Reading Custom Dataset===================
winsList = os.listdir("sub/")
custom = {}
for w in winsList:
    df = pd.read_excel("sub/"+w)
    cat = df["categories"].unique()
    subset = {}
    for c in cat:
        subset[c] = df[df["categories"] == c]["questions"].to_list()
    custom[w[:-5]] = subset
#==============================

# names that has to be excluded
outNames = ["android","mac","linux","ubantu","watch"]

for var in winName:
    globals()[var] = pd.read_excel(f"data/{var}.xlsx")
    globals()[var+"_custom"] = pd.read_excel(f"data/custom/{var}.xlsx")

for var in winName:
    winProblems[var] = eval(var+"_custom")["Problems"].to_list()

# cleaning text and standardising
def clean_texts(statements):
    non_punch = ''
    for i in statements:
        if i not in string.punctuation:
            non_punch += i
    split_text = re.split('\W+',non_punch)
    stop = nltk.corpus.stopwords.words('english')
    word_lst = []
    for i in split_text:
        if i not in stop:
            word_lst.append(i)
    wn = nltk.WordNetLemmatizer()
    clean_sen = ""
    for statement in word_lst:
        clean_sen = clean_sen +  wn.lemmatize(statement).lower()+" "
    return clean_sen.strip()


for d in winName:
    globals()[d+"d"] = eval(d)["Problems"].apply(lambda x: clean_texts(x))
    globals()[d+"d_custom"] = eval(d)["Problems"].apply(lambda x: clean_texts(x))

def checkOutNames(sent):
    for name in outNames:
        if name.strip().lower() in sent:
            return (True,name)
    else:
        return (False,"no")

@app.route("/")
def home():
    return render_template("home.html",winNames=winName,winProbelm=winProblems,subpro=custom)
    

@app.route("/ans")
def getMyAns():
    userText = request.args.get('msg')
    osType = request.args.get('osType')
    key = request.args.get('cat')

    out = checkOutNames(userText)
    if out[0]:
        return "sorry no data related to "+out[1]+" available"

    try:
        data = pd.read_excel("sub/"+osType+".xlsx")
        ans = data[data["questions"] == userText]["answers"].to_list()
        if ans == []:
            return  "No Answer found"
        else:
            return ans[0]
    except:
        return "something went wrong please check the sub/ folder"

@app.route("/get")
def get_bot_response():
    type = request.args.get('type')
    userText = request.args.get('msg')

    out = checkOutNames(userText) # checking if the input text is not out of context

    if str(userText).strip() in swords:
        return ""   
    if out[0]:
        return "sorry no data related to "+out[1]+" available"

    # applying model to determining the type of OS the text belong
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([clean_texts(userText)]),
                                                                      truncating='post', maxlen=max_len))
    # getting the closest sentence to user input text and returning the respective answer from the dataset
    try:
        osType = wins[np.argmax(result)]
        sim = []
        win = eval(osType)
        for item in eval(osType+"d").to_list():
            sim.append(fuzz.ratio(clean_texts(userText), item))
            hig = sim.index(sorted(sim)[len(sim)-int(type)])
        return 	win["Solution"][hig]+" solution from "+osType
    except:
        return "No Answer found 😌"

if __name__ == "__main__":
    app.run(debug=True,port=1000)

