from flask import Flask, render_template, json, request
import json
import traceback
import os


from threading import Lock
lock = Lock()


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/init', methods=['POST'])
def init():
    html = "_init_"
    return html


#nnet api
import api

@app.route('/show_content', methods=['POST'])
def show_content():
    data = json.loads(request.data)
    prev_fname = data[u'prev_state_filename']
    n_top_qvalues = 5


    if os.path.isfile(prev_fname) :
        answer = ""

        try:
            response = int(data[u'response']) !=0
            result = api.get_next_state(prev_fname, response)
            new_fname,qvalues,action = result
            answer += "New state:<br>%s<br><br>"%new_fname

        except:
            answer += "Could not read response. Has to be 1 or 0 in this setup. <br> Showing state instead.<br>"
            new_fname,qvalues,action = api.load_state(prev_fname)


        
        answer += "Agent action: "+action+"<br>"
        answer += "<br>Top-%i Qvalues:<br>" % n_top_qvalues
        
        
        for a, qv in sorted(qvalues.items(),key=lambda x: -x[1])[:n_top_qvalues]:
            answer += "%s : %.5f<br>"%(a,qv)
    else:
        result = api.new_session()
        new_fname,qvalues,action = result
        answer = "Setting up new session... <br><br>"
        answer += "Agent action: "+action+"<br>"
        answer += "New state:<br>%s<br><br>"%new_fname
        answer += "<br>Top-%i Qvalues:<br>" % n_top_qvalues
        
        
        for a, qv in sorted(qvalues.items(),key=lambda x: -x[1])[:n_top_qvalues]:
            answer += "%s : %.5f<br>"%(a,qv)
        
        
    print json.dumps({"ans":"<p>"+answer+"</p>", "fname":new_fname})
    with lock:
        return json.dumps({"ans":"<p>"+answer+"</p>", "fname":new_fname})


if __name__ == '__main__':
    print "demo ready"

    app.run(debug=True, host='0.0.0.0')
