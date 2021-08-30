# Machine_Translation


## Members:

Moran Oshia 

Alex Chagan

Raam Banin



## **Purpose**

In this document we are going to describe how our final project in the subject of “Machine Translation” is going to work at the course “Deep Learning Methods for Natural Language Processing & Speech Recognition”, a little bit on the algorithm of seq2seq,preparations of the data sets, how we used OOP principle, how to train the models and the user interface and how to use it. Our main goal is to program a user-friendly GUI web application to translate a word or sentences with the seq2seq model after training.

**Short explanation about seq2seq**

The most common sequence-to-sequence (seq2seq) models are encoder-decoder models, which commonly use a recurrent neural network (RNN) to encode the source (input) sentence into a single vector.We can think of the context vector as being an abstract representation of the entire input sentence.This vector is then decoded by a second RNN which learns to output the target (output) sentence by generating it one word at a time.

## **How does it work?**

The idea being that you get one model encoder to take a sequence of words and turn them into an encoder vector which represents all the information in the sentence you want to translate. Then you get a decoder that takes this vector and outputs the translated words using a softmax function.

## **Attention Layers**

The attention layer allows the decoder to focus on the parts of the sequence needed for getting the output correct at that time step, therefore shortening the path between the input word to its translation, thereby alleviating some of the memory limitations that LSTMs can have. It does this by providing a way of scoring tokens in the target sequence against all the tokens on the source sequence and using this to change the input to the decoder sequence. The scores are then fed into a softmax activation to make an attention distribution.



## **Requirements**

● Python 3.8

● Ubuntu

● torch== 1.9.0

● numpy==1.21.1

● levenshtein~=0.12.0

● Flask~=2.0.0

● matplotlib~=3.4.2

You can run the command in the terminal
$ pip install -r requirements.txt
and it will install the necessary packages.



## **Creating the model step by step**

The model is created with the help of Google’s Colab notebook:

<https://research.google.com/colaboratory>

1. Connect to GPU:

        Runtime -> change runtime type -> GPU -> save.

2. clone Github. activation of the folder by name Machine\_Translation:

        !git clone https://github.com/MoranOshia/Machine\_Translation

        %cd Machine\_Translation

3. Install requirements: Installs versions of torch and numpy:

        !pip install -r requirements.txt

4. Pre-process

        global vars:

        *lang1* and *lang2* choosing languages for the translation.

        reverse switches between lang1 to lang2

        example: lang1=’eng’, lang2=’fra’, reverse=True

        The translation will be from French to english.

        Activating Python code: create dictionary pickle

        **!python /content/Machine\_Translation/pre\_process.py**

5. Training

        global vars:

        *Dictionary name:* pickle to dictionary. Write the name of the pickle we want.

        example: "eng-fra-dictionary.pickle"

        Activating Python code: create model pickle

        **!python /content/Machine\_Translation/training.py**

6. Inference

        global vars:

        *Model name:* pickle to model. Write the name of the pickle we want.

        example: "eng-fra-model.pickle"

        Activating Python code: test the training model

        **!python /content/Machine\_Translation/inference.py**





## **User Interface**

We decided to use the Flask for our Front-End in order to connect between the HTML and the Python, by using the Flask we can run the device on localhost.

Install and activate Flask:

1. At first we installed the Flask package by using the following command :

        $ pip install Flask
  
     (Now it is in the requirement.txt included)

2. To run the application, use the flask command or python -m flask. Before you can do that you need to tell your terminal the application to work with by exporting the FLASK_APP environment variable, by using the following command :

         $ export FLASK_APP=flask_MT.py

         $ flask run

3. To enable all DEBUG features, set the FLASK\_DEBUG environment variable to 1, by using the following command:

         $ export FLASK\_DEBUG=1

Reference Material:

<https://flask.palletsprojects.com/en/2.0.x/installation/#python-version>
<https://flask.palletsprojects.com/en/2.0.x/quickstart/#a-minimal-application>

### **Running the server in localhost:**

Now, you can simply use the following command in order to run the server in localhost

http://127.0.0.1 in port 5000:

        $ python flask_MT.py


![alt text](https://github.com/MoranOshia/Machine_Translation/blob/main/images/image/open_localhost.PNG)



## **How to use the web application:**

After running the localhost and entering the URL <http://127.0.0.1:5000>[ ](http://127.0.0.1:5000)in browser, it will

open the web in the main page:

![alt text](https://github.com/MoranOshia/Machine_Translation/blob/main/images/image/main_page_ints.png)

## **In order to translate a word or a sentence you will need:**

● Choose the languages you want to translate from to other language in the drop down list (1).

● Enter the sentence from the origin language you want to translate (2).

● Press on the button Translate (3).





### **Example of translation:**


![alt text](https://github.com/MoranOshia/Machine_Translation/blob/main/images/image/translate_1.png)


### **After clicking on “Translate” button:**


![alt text](https://github.com/MoranOshia/Machine_Translation/blob/main/images/image/translate_2.png)

