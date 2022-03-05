import flask
import pandas as pd


# Use pickle to load in the pre-trained model.
# import pickle 
# with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
#     model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])

def main():
    print("in main")
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))


    if flask.request.method == 'POST':
        print("AT POST")
        moral_question = flask.request.form['moral question']
        print("moral question", moral_question)

        prediction = 23 
        return flask.render_template('main.html',original_input=moral_question,result=prediction, )
    
if __name__ == '__main__':
    app.run()

