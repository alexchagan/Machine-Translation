import torch

# Shared vars

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_device_inf = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_hidden_size = 256

_MAX_LENGTH = 10

_SOS_token = 0

_EOS_token = 1

# Languages

_lang1 = 'heb'

_lang2 = 'arm'

_reverse = True

# Training vars

if _reverse:
    _dictionary_name = 'dictionaries/'+_lang2+'-'+_lang1+'-dictionary.pickle'
else:
    _dictionary_name = 'dictionaries/'+_lang1 + '-' + _lang2 + '-dictionary.pickle'

_teacher_forcing_ratio = 0.5

_n_iters = 1000

_print_every= 500

_plot_every=100

_learning_rate=0.01

_dropout_p=0.1

_load_pickle = False

_pickle_name = ""

# Inference vars

if _reverse:
    _model_name =  'model/'+_lang2+'-'+_lang1+'-model.pickle'
else:
    _model_name =  'model/' + _lang1 + '-' + _lang2 + '-model.pickle'

