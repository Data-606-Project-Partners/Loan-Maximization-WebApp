from flask import Flask, render_template, request, make_response, jsonify
from waitress import serve

from models import defaultOrNot, maximizeLoan

import pickle
class_model = ""
regress_model = ""
encoding = ""


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',)

@app.route('/model',methods= ['POST'])
def returnModel():
    data = request.get_json()

    #for key in data.keys():
    #    print (f'{key}: {data[key]}')
    
    results = defaultOrNot( data , encoding, class_model )


    if ( results[0] == -1):
        answer = make_response( jsonify({'status' : 'Invalid Input'}), 200)
    elif ( results[0] == False ):
        answer = make_response( jsonify({'status': 'Fully Paid'}), 200)
    else:
        loss,meth1,meth2 = maximizeLoan(results[1], regress_model)

        answer = make_response( jsonify( {'status': 'Defaulted', 'loss': f'{loss}', 'sugg1': f'{meth1}', 'sugg2' : f'{meth2}' }), 200)

    return answer


def load():
    with open('Step_1_Model.pickle', 'rb') as model1:
        global class_model
        class_model = pickle.load(model1)

    with open('Step_2_Model.pickle', 'rb') as model2:
        global regress_model
        regress_model = pickle.load(model2)

    with open('label_encoder.pickle', 'rb') as encoder:
        global encoding
        encoding = pickle.load(encoder)





if __name__ == "__main__":
    if (class_model == "" or regress_model == "" or encoding == ""):
        load()
        print("Server Active")

    serve(app, host = "0.0.0.0", port = 5000)




    