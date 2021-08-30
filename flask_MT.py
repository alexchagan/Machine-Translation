from flask import Flask, render_template, url_for
from flask import request
from pickle_funcs import load_pickle
from pre_process import PreProcessData,Lang
from training import TrainingObject,train_obj
from models import EncoderRNN,AttnDecoderRNN
from algos import get_choosen_languages,translate,closest_sentence,display_lev_changes


app = Flask(__name__,template_folder='Templates')
@app.route("/", methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])
def home():
	choosen_lan = ''
	result = ''
	lev_changes = ''
	changes = ''

	if request.method == 'POST':
		choosen_lan = request.form.get('Lang')
		lang1 , lang2 = get_choosen_languages(choosen_lan)
		origin_words = []
		new_words = [] #after use the lev algo
		changes = ""
		pickle_model_name ='model/'+lang1+'-'+lang2+'-model.pickle'
		pickle_dict_name = 'dictionaries/'+lang1+'-'+lang2+'-dictionary.pickle'
		sen = request.form.get('sen')
		sen , new_words = closest_sentence(sen,pickle_dict_name)
		lev_changes = "Changes after Levenshtein: \n"
		changes = display_lev_changes(new_words)
		print(changes)
		result=translate(pickle_dict_name,pickle_model_name,sen)
        	#result = evaluate(pickle_dict,encoder,decoder,sen)
		return render_template('index.html', word=result,lev=lev_changes,changes=changes,choosen_lan=choosen_lan)

	if request.method == 'GET':
		return render_template('index.html', word=result, lev=lev_changes, changes=changes,choosen_lan=choosen_lan)

	return render_template('index.html', word="")


@app.route("/about", methods=['GET','POST'])
def about():

	return render_template('about.html')

if __name__ == '__main__':
	app.run(debug=True)
