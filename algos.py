import Levenshtein as lev
from pre_process import PreProcessData,Lang
from training import TrainingObject
from models import EncoderRNN,AttnDecoderRNN
from pickle_funcs import load_pickle
from inference import evaluate
import numpy as np



def levenshtein_distance(s, words):
    min_distance = 1000
    best_word = ""

    for i in range(len(words)):
        distance = lev.distance(s, words[i])
        if distance < min_distance:
            best_word = words[i]
            min_distance = distance
    return best_word


def closest_sentence(sent, dictionary_name):

    dictionary = load_pickle(dictionary_name)
    words = sent.split(" ")

    temp = dictionary.input_lang.word2index
    temp = list(temp)
    dictionary_words = np.asarray(temp)

    new_words = []
    ui_origin_words = []    
    ui_replace_words = []
    for i in range(len(words)):
        if words[i] in dictionary_words:
            new_words.append(words[i])
        else:
        
            new = levenshtein_distance(words[i], dictionary_words)
            new_words.append(new)
            #ui_origin_words.append((words[i],new))
            ui_replace_words.append((words[i],new))           

    new_sent = ''
    for i in range(len(words)-1):
        new_sent += new_words[i]
        new_sent += ' '
    new_sent +=new_words[len(words)-1]	
    return new_sent, ui_replace_words


def translate(dict_name,model_name,sen):

    pickle_dict=load_pickle(dict_name)
    pickle_model = load_pickle(model_name)
    encoder = pickle_model.encoder
    decoder = pickle_model.decoder
    sen_trans,tesn = evaluate(pickle_dict,encoder,decoder,sen)
    tran_sen=''
    l = len(sen_trans)
    for i in range(l):
        if not sen_trans[i]=='<EOS>':
            tran_sen=tran_sen+' '+sen_trans[i]
            

    return tran_sen

def get_choosen_languages(s):
    lang1 = ''
    lang2 = ''

    if s == 'En2Fr':
        lang1 = 'eng'
        lang2 = 'fra'

    if s == 'Fr2En':
        lang1 = 'fra'
        lang2 = 'eng'

    if s == 'He2Ar':
        lang1 = 'heb'
        lang2 = 'arm'

    if s == 'Ar2He':
        lang1 = 'arm'
        lang2 = 'heb'

    return lang1, lang2

def display_lev_changes(new_words):

    len_words=len(new_words)
    s = []
    new=""
    origin=""
    if len_words == 0 :
    	s.append("There is no changes in the sentence")
    else:
    	for i in range(len_words):
    		new = new_words[i][0]
    		origin = new_words[i][1]
    		print("tttttttttttttt ", new, origin)
    		s.append("Origin Word: "+ new + " || New Word: " + origin)
    return s
    		
    		 
    		 
    
