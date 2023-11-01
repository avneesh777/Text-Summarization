from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QStackedWidget, QFileDialog
import pandas as pd
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
import time
from tensorflow.python.client import timeline

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

from keras import backend as K 
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Global Variable declaration:
# Dictionary to convert words to integers
vocab_to_int = {} 
sorted_texts = 0
batch_size = 0
sorted_summaries = 0
# Set the Hyperparameters
epochs = 2 # use 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75
train_op = 0
cost = 0
input_data = 0
targets = 0
lr = 0
summary_length = 0
text_length = 0
keep_prob = 0
train_graph = 0
word_embedding_matrix = 0




class Ui_Data_processing(QDialog):
    all_data = None
    def __init__(self, all_data):
        super(Ui_Data_processing, self).__init__()
        loadUi(r"src\data_processing.ui", self)

        # path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')[0]

        # self.all_data = pd.read_csv("Reviews\Reviews.csv")


        csv_file_path = 'D:/Internet_Downloads/Programming_Languages/Programs/Text_Summarization/Reviews/Reviews.csv'

        # Read the CSV file
        self.all_data = pd.read_csv(csv_file_path)


        print("\n\n\n")
        print("#############################################################################")
        print(self.all_data.head())
        print("\n\n\n")
        print("Data loaded successfully!")
        print("#############################################################################")
        print("\n\n\n")

        self.Text_Preprocessing.clicked.connect(self.Text_Preprocessing_Data)
        self.Model_Building.clicked.connect(self.Model_Building_Data)
        # self.Training_Model.clicked.connect(self.Training_Model_Data)



    def Text_Preprocessing_Data(self):
        global vocab_to_int, sorted_texts, batch_size, sorted_summaries, train_op, cost, input_data, lr, summary_length, keep_prob, train_graph, word_embedding_matrix


        # Check for any nulls values
        print("\n\n\n")
        print("#############################################################################")
        print("Check for any nulls values:")
        print(self.all_data.isnull().sum())
        print("#############################################################################")
        print("\n\n\n")


        

        # Check for any nulls values
        print("\n\n\n")
        print("#############################################################################")
        print("Remove null values and unneeded features:")
        # Remove null values and unneeded features
        self.all_data = self.all_data.dropna()
        self.all_data = self.all_data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time'], axis=1)
        self.all_data = self.all_data.reset_index(drop=True)
        print("#############################################################################")
        print("\n\n\n")


        # Checking dataframe status after above two operations
        print("\n\n\n")
        print("#############################################################################")
        print("Checking dataframe status after above two operations:")
        print(self.all_data.head())
        print("#############################################################################")
        print("\n\n\n")


        # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are"
        }



        def clean_text(text, remove_stopwords = True):
            # Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings
            
            # Convert words to lower case
            text = text.lower()
            
            # Replace contractions with their longer forms 
            if True:
                text = text.split()
                new_text = []
                for word in text:
                    if word in self.contractions:
                        new_text.append(self.contractions[word])
                    else:
                        new_text.append(word)
                text = " ".join(new_text)
            
            # Format words and remove unwanted characters
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\<a href', ' ', text)
            text = re.sub(r'&amp;', '', text) 
            text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
            text = re.sub(r'<br />', ' ', text)
            text = re.sub(r'\'', ' ', text)
            
            # Optionally, remove stop words
            if remove_stopwords:
                text = text.split()
                stops = set(stopwords.words("english"))
                text = [w for w in text if not w in stops]
                text = " ".join(text)

            return text
        

        # Checking dataframe status after above two operations
        print("\n\n\n")
        print("#############################################################################")
        print("We will remove the stopwords from the texts because they do not provide much use for training our model. However, we will keep them for our summaries so that they sound more like natural phrases:")
        print("#############################################################################")
        print("\n\n\n")
                
        # Clean the summaries and texts
        clean_summaries = []
        for summary in self.all_data.Summary:
            clean_summaries.append(clean_text(summary, remove_stopwords=False))
        print("Summaries are complete.")

        clean_texts = []
        for text in self.all_data.Text:
            clean_texts.append(clean_text(text, remove_stopwords=False))
        print("Texts are complete.")

        print("\n\n\n")
        print("#############################################################################")
        print("\n\n\n")

        print("\n\n\n")
        print("#############################################################################")
        print("Clean the summaries and texts:")
        print("#############################################################################")
        print("\n\n\n")
        # Inspect the cleaned summaries and texts to ensure they have been cleaned well
        for i in range(5):
            print("Clean Review #",i+1)
            print(clean_summaries[i])
            print(clean_texts[i])
            print()


        def count_words(count_dict, text):
            # Count the number of occurrences of each word in a set of text
            for sentence in text:
                for word in sentence.split():
                    if word not in count_dict:
                        count_dict[word] = 1
                    else:
                        count_dict[word] += 1



        # Find the number of times each word was used and the size of the vocabulary
        word_counts = {}

        count_words(word_counts, clean_summaries)
        count_words(word_counts, clean_texts)
        print("\n\n\n")
        print("#############################################################################")
        print("Find the number of times each word was used and the size of the vocabulary:")
        print("#############################################################################")
        print("\n\n\n")            
        print("Size of Vocabulary:", len(word_counts))


        embeddings_index = []

        # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better (https://github.com/commonsense/conceptnet-numberbatch)
        embeddings_index = {}
        with open(r"numberbatch-en-17.02.txt\numberbatch-en-17.02.txt", encoding='utf-8') as f:
           for line in f:
               values = line.split(' ')
               word = values[0]
               embedding = np.asarray(values[1:], dtype='float32')
               embeddings_index[word] = embedding
        print("\n\n\n")
        print("#############################################################################")
        print('Word embeddings:', len(embeddings_index))
        print("#############################################################################")
        print("\n\n\n")     
        


        # Find the number of words that are missing from CN, and are used more than our threshold.
        missing_words = 0
        threshold = 20

        for word, count in word_counts.items():
            if count > threshold:
                if word not in embeddings_index:
                    missing_words += 1
                    
        missing_ratio = round(missing_words/len(word_counts),4)*100

        print("\n\n\n")
        print("#############################################################################")
        print("Number of words missing from CN:", missing_words)
        print("\n\n\n")
        print("#############################################################################")
        print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))
        print("#############################################################################")
        print("\n\n\n")  



        # I use a threshold of 20, so that words not in CN can be added to our word_embedding_matrix, but they need to be common enough in the reviews so that the model can understand their meaning.
        # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

        
        value = 0
        for word, count in word_counts.items():
            if count >= threshold or word in embeddings_index:
                vocab_to_int[word] = value
                value += 1

        # Special tokens that will be added to our vocab
        codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

        # Add codes to vocab
        for code in codes:
            vocab_to_int[code] = len(vocab_to_int)

        # Dictionary to convert integers to words
        int_to_vocab = {}
        for word, value in vocab_to_int.items():
            int_to_vocab[value] = word

        usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

        print("\n\n\n")
        print("#############################################################################")
        print("Total number of unique words:", len(word_counts))
        print("\n\n\n")

        print("#############################################################################")
        print("Number of words we will use:", len(vocab_to_int))

        print("\n\n\n")
        print("#############################################################################")
        print("Percent of words we will use: {}%".format(usage_ratio))
        
        print("#############################################################################")
        print("\n\n\n")    



        # Need to use 300 for embedding dimensions to match CN's vectors.
        embedding_dim = 300
        nb_words = len(vocab_to_int)

        # Create matrix with default values of zero
        word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
        for word, i in vocab_to_int.items():
            if word in embeddings_index:
                word_embedding_matrix[i] = embeddings_index[word]
            else:
                # If word not in CN, create a random embedding for it
                new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
                #embeddings_index[word] = new_embedding
                word_embedding_matrix[i] = new_embedding

        # Check if value matches len(vocab_to_int)
        print("\n\n\n")
        print("#############################################################################")
        print("Check if value matches len(vocab_to_int)")
        print(len(word_embedding_matrix))
        
        print("#############################################################################")
        print("\n\n\n")  

        def convert_to_ints(text, word_count, unk_count, eos=False):
            '''Convert words in text to an integer.
            If word is not in vocab_to_int, use UNK's integer.
            Total the number of words and UNKs.
            Add EOS token to the end of texts'''
            ints = []
            for sentence in text:
                sentence_ints = []
                for word in sentence.split():
                    word_count += 1
                    if word in vocab_to_int:
                        sentence_ints.append(vocab_to_int[word])
                    else:
                        sentence_ints.append(vocab_to_int["<UNK>"])
                        unk_count += 1
                if eos:
                    sentence_ints.append(vocab_to_int["<EOS>"])
                ints.append(sentence_ints)
            return ints, word_count, unk_count
        

        # Apply convert_to_ints to clean_summaries and clean_texts
        word_count = 0
        unk_count = 0

        int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
        int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

        unk_percent = round(unk_count/word_count,4)*100

        print("\n\n\n")
        print("#############################################################################")
        print("Total number of words in headlines:", word_count)

        print("#############################################################################")
        print("\n\n\n") 
        print("Total number of UNKs in headlines:", unk_count)
        
        print("#############################################################################")
        print("\n\n\n")        
        print("Percent of words that are UNK: {}%".format(unk_percent)) 

        
        

        def create_lengths(text):
            '''Create a data frame of the sentence lengths from a text'''
            lengths = []
            for sentence in text:
                lengths.append(len(sentence))
            return pd.DataFrame(lengths, columns=['counts'])
        
        lengths_summaries = create_lengths(int_summaries)
        lengths_texts = create_lengths(int_texts)

        print("#############################################################################")
        print("\n\n\n") 
        print("Summaries:")
        print(lengths_summaries.describe())
        print("#############################################################################")
        print("\n\n\n") 
        print("#############################################################################")
        print("\n\n\n") 
        print("Texts:")

        print("#############################################################################")
        print("\n\n\n") 
        print(lengths_texts.describe())


        # Inspect the length of texts
        print("Inspect the length of texts")
        print("#############################################################################")
        print(np.percentile(lengths_texts.counts, 90))
        print(np.percentile(lengths_texts.counts, 95))
        print(np.percentile(lengths_texts.counts, 99))
        print("\n\n\n")

        
        # Inspect the length of summaries
        print("Inspect the length of summaries")
        print("#############################################################################")
        print(np.percentile(lengths_summaries.counts, 90))
        print(np.percentile(lengths_summaries.counts, 95))
        print(np.percentile(lengths_summaries.counts, 99))
        print("\n\n\n") 

        def unk_counter(sentence):
            '''Counts the number of time UNK appears in a sentence.'''
            unk_count = 0
            for word in sentence:
                if word == vocab_to_int["<UNK>"]:
                    unk_count += 1
            return unk_count
        

        # Sort the summaries and texts by the length of the texts, shortest to longest
        # Limit the length of summaries and texts based on the min and max ranges.
        # Remove reviews that include too many UNKs
        
        sorted_summaries = []
        sorted_texts = []
        max_text_length = 84
        max_summary_length = 13
        min_length = 2
        unk_text_limit = 100 # use 1
        unk_summary_limit = 100 # use 0


        for length in range(min(lengths_texts.counts), max_text_length): 
            for count, words in enumerate(int_summaries):
                if (len(int_summaries[count]) >= min_length and
                    len(int_summaries[count]) <= max_summary_length and
                    len(int_texts[count]) >= min_length and
                    unk_counter(int_summaries[count]) <= unk_summary_limit and
                    unk_counter(int_texts[count]) <= unk_text_limit and
                    length == len(int_texts[count])
                ):
                    sorted_summaries.append(int_summaries[count])
                    sorted_texts.append(int_texts[count])
                
        
        # Compare lengths to ensure they match
        print("Comparing lengths for checking values match")
        print("#############################################################################")
        print(len(sorted_summaries))
        print(len(sorted_texts))
        print("\n\n\n") 


        self.Text_Preprocessing.setEnabled(False)


    def Model_Building_Data(self):
            global batch_size
            # def model_inputs():
            #     '''Create palceholders for inputs to the model'''
                
            #     input_data = tf.compat.v1.placeholder(tf.int32, [None, None], name='input')
            #     # input_data = tf.Variable(tf.int32, [None, None], name='input')
                
            #     # targets = tf.placeholder(tf.ones(tf.int32, shape=[None, None], name='targets'))
            #     targets = tf.compat.v1.placeholder(tf.int32, [None, None], name='targets')


            #     lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
            #     keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
            #     summary_length = tf.compat.v1.placeholder(tf.int32, (None,), name='summary_length')
            #     max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
            #     text_length = tf.placeholder(tf.int32, (None,), name='text_length')

            #     return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length
            

            # def process_encoding_input(target_data, vocab_to_int, batch_size):
            #     '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
            #     ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
            #     dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

            #     return dec_input


            # def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
            #     '''Create the encoding layer'''
                
            #     for layer in range(num_layers):
            #         with tf.variable_scope('encoder_{}'.format(layer)):
            #             cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
            #                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #             cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
            #                                                     input_keep_prob = keep_prob)

            #             cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
            #                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #             cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
            #                                                     input_keep_prob = keep_prob)

            #             enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
            #                                                                     cell_bw, 
            #                                                                     rnn_inputs,
            #                                                                     sequence_length,
            #                                                                     dtype=tf.float32)

            #             # cell_fw = tf.keras.layers.LSTMCell(rnn_size, kernel_initializer='random_uniform')
            #             # cell_fw = tf.keras.layers.Dropout(1 - keep_prob)(cell_fw)

            #             # cell_bw = tf.keras.layers.LSTMCell(rnn_size, kernel_initializer='random_uniform')
            #             # cell_bw = tf.keras.layers.Dropout(1 - keep_prob)(cell_bw)


            #             # enc_output, forward_state, backward_state = tf.keras.layers.Bidirectional(
            #             #     cell_fw=cell_fw, cell_bw=cell_bw, dtype=tf.float32
            #             # )(rnn_inputs, sequence_length=sequence_length)
            #             # enc_state = [forward_state, backward_state]

            #     # Join outputs since we are using a bidirectional RNN
            #     enc_output = tf.concat(enc_output, 2)
                
            #     return enc_output, enc_state
            

            # def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
            #                 vocab_size, max_summary_length):
            #     '''Create the training logits'''
                
            #     training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
            #                                                         sequence_length=summary_length,
            #                                                         time_major=False)

            #     training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
            #                                                     training_helper,
            #                                                     initial_state,
            #                                                     output_layer) 

            #     training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
            #                                                         output_time_major=False,
            #                                                         impute_finished=True,
            #                                                         maximum_iterations=max_summary_length)
            #     return training_logits
            


            # def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
            #                  max_summary_length, batch_size):
            #     '''Create the inference logits'''
                
            #     start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
                
            #     inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
            #                                                                 start_tokens,
            #                                                                 end_token)
                            
            #     inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
            #                                                         inference_helper,
            #                                                         initial_state,
            #                                                         output_layer)
                            
            #     inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
            #                                                             output_time_major=False,
            #                                                             impute_finished=True,
            #                                                             maximum_iterations=max_summary_length)
                
            #     return inference_logits
            

            # def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
            #        max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
            #         '''Create the decoding cell and attention for the training and inference decoding layers'''
                    
            #         for layer in range(num_layers):
            #             with tf.variable_scope('decoder_{}'.format(layer)):
            #                 lstm = tf.contrib.rnn.LSTMCell(rnn_size,
            #                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #                 dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
            #                                                         input_keep_prob = keep_prob)
                    
            #         output_layer = Dense(vocab_size,
            #                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
                    
            #         attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
            #                                                     enc_output,
            #                                                     text_length,
            #                                                     normalize=False,
            #                                                     name='BahdanauAttention')

            #         dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
            #                                                             attn_mech,
            #                                                             rnn_size)
                            
            #         initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],
            #                                                                         _zero_state_tensors(rnn_size, 
            #                                                                                             batch_size, 
            #                                                                                             tf.float32)) 
            #         with tf.variable_scope("decode"):
            #             training_logits = training_decoding_layer(dec_embed_input, 
            #                                                     summary_length, 
            #                                                     dec_cell, 
            #                                                     initial_state,
            #                                                     output_layer,
            #                                                     vocab_size, 
            #                                                     max_summary_length)
            #         with tf.variable_scope("decode", reuse=True):
            #             inference_logits = inference_decoding_layer(embeddings,  
            #                                                         vocab_to_int['<GO>'], 
            #                                                         vocab_to_int['<EOS>'],
            #                                                         dec_cell, 
            #                                                         initial_state, 
            #                                                         output_layer,
            #                                                         max_summary_length,
            #                                                         batch_size)

            #         return training_logits, inference_logits
            
            # def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
            #       vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
            #         '''Use the previous functions to create the training and inference logits'''
                    
            #         # Use Numberbatch's embeddings and the newly created ones as our embeddings
            #         embeddings = word_embedding_matrix
                    
            #         enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
            #         enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
                    
            #         dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
            #         dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
                    
            #         training_logits, inference_logits  = decoding_layer(dec_embed_input, 
            #                                                             embeddings,
            #                                                             enc_output,
            #                                                             enc_state, 
            #                                                             vocab_size, 
            #                                                             text_length, 
            #                                                             summary_length, 
            #                                                             max_summary_length,
            #                                                             rnn_size, 
            #                                                             vocab_to_int, 
            #                                                             keep_prob, 
            #                                                             batch_size,
            #                                                             num_layers)
                    
            #         return training_logits, inference_logits
            


            # def pad_sentence_batch(sentence_batch, vocab_to_int):
            #     """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
            #     max_sentence = max([len(sentence) for sentence in sentence_batch])
            #     return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
            
            # def get_batches(summaries, texts, batch_size):
            #     """Batch summaries, texts, and the lengths of their sentences together"""
            #     for batch_i in range(0, len(texts)//batch_size):
            #         start_i = batch_i * batch_size
            #         summaries_batch = summaries[start_i:start_i + batch_size]
            #         texts_batch = texts[start_i:start_i + batch_size]
            #         pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
            #         pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
                    
            #         # Need the lengths for the _lengths parameters
            #         pad_summaries_lengths = []
            #         for summary in pad_summaries_batch:
            #             pad_summaries_lengths.append(len(summary))
                    
            #         pad_texts_lengths = []
            #         for text in pad_texts_batch:
            #             pad_texts_lengths.append(len(text))
                    
            #         yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


            # get_batches(sorted_summaries, sorted_texts, batch_size)


            # # Set the Hyperparameters
            # epochs = 2 # use 100
            # batch_size = 64
            # rnn_size = 256
            # num_layers = 2
            # learning_rate = 0.005
            # keep_probability = 0.75


            # # Build the graph
            # train_graph = tf.Graph()
            # # Set the graph to default to ensure that it is ready for training
            # with train_graph.as_default():
                
            #     # Load the model inputs    
            #     input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

            #     # Create the training and inference logits
            #     training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
            #                                                     targets, 
            #                                                     keep_prob,   
            #                                                     text_length,
            #                                                     summary_length,
            #                                                     max_summary_length,
            #                                                     len(vocab_to_int)+1,
            #                                                     rnn_size, 
            #                                                     num_layers, 
            #                                                     vocab_to_int,
            #                                                     batch_size)
                
            #     # Create tensors for the training logits and inference logits
            #     training_logits = tf.identity(training_logits.rnn_output, 'logits')
            #     inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
                
            #     # Create the weights for sequence_loss
            #     masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

            #     with tf.name_scope("optimization"):
            #         # Loss function
            #         cost = tf.contrib.seq2seq.sequence_loss(
            #             training_logits,
            #             targets,
            #             masks)

            #         # Optimizer
            #         optimizer = tf.train.AdamOptimizer(learning_rate)

            #         # Gradient Clipping
            #         gradients = optimizer.compute_gradients(cost)
            #         capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            #         train_op = optimizer.apply_gradients(capped_gradients)
            # print("Graph is built.")




            K.clear_session()

            latent_dim = 300
            embedding_dim=100

            # Encoder
            encoder_inputs = Input(shape=(max_text_len,))

            #embedding layer
            enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)

            #encoder lstm 1
            encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
            encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

            #encoder lstm 2
            encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
            encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

            #encoder lstm 3
            encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
            encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

            # Set up the decoder, using `encoder_states` as initial state.
            decoder_inputs = Input(shape=(None,))

            #embedding layer
            dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
            dec_emb = dec_emb_layer(decoder_inputs)

            decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
            decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

            # Attention layer
            attn_layer = AttentionLayer(name='attention_layer')
            attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

            # Concat attention input and decoder LSTM output
            decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

            #dense layer
            decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))
            decoder_outputs = decoder_dense(decoder_concat_input)

            # Define the model 
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

            model.summary() 

            
            self.Model_Building.setEnabled(False)

    def Training_Model_Data(self):
        global sorted_texts, batch_size
        # Train the Model
        learning_rate_decay = 0.95
        min_learning_rate = 0.0005
        display_step = 20 # Check training loss after every 20 batches
        stop_early = 0 
        stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
        per_epoch = 3 # Make 3 update checks per epoch
        update_check = (len(sorted_texts)//batch_size//per_epoch)-1

        update_loss = 0 
        batch_loss = 0
        summary_update_loss = [] # Record the update losses for saving improvements in the model

        checkpoint = "best_model.ckpt" 
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            # If we want to continue training a previous session
            # loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
            # loader.restore(sess, checkpoint)
            
            for epoch_i in range(1, epochs+1):
                update_loss = 0
                batch_loss = 0
                for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                        self.Model_Building_Data(sorted_summaries, sorted_texts, batch_size)):
                    start_time = time.time()
                    _, loss = sess.run(
                        [train_op, cost],
                        {input_data: texts_batch,
                        targets: summaries_batch,
                        lr: learning_rate,
                        summary_length: summaries_lengths,
                        text_length: texts_lengths,
                        keep_prob: keep_probability})

                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time

                    if batch_i % display_step == 0 and batch_i > 0:
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                            .format(epoch_i,
                                    epochs, 
                                    batch_i, 
                                    len(sorted_texts) // batch_size, 
                                    batch_loss / display_step, 
                                    batch_time*display_step))
                        batch_loss = 0

                    if batch_i % update_check == 0 and batch_i > 0:
                        print("Average loss for this update:", round(update_loss/update_check,3))
                        summary_update_loss.append(update_loss)
                        
                        # If the update loss is at a new minimum, save the model
                        if update_loss <= min(summary_update_loss):
                            print('New Record!') 
                            stop_early = 0
                            saver = tf.train.Saver() 
                            saver.save(sess, checkpoint)

                        else:
                            print("No Improvement.")
                            stop_early += 1
                            if stop_early == stop:
                                break
                        update_loss = 0
                    
                            
                # Reduce learning rate, but not below its minimum value
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate
                
                if stop_early == stop:
                    print("Stopping Training.")
                    break
        

                




if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv) # Launching the app with this variable
    all_data = pd.read_csv(r"Reviews\Reviews.csv")
    initial_screen = Ui_Data_processing(all_data) # Creating an instance for the class created above

    widget = QStackedWidget() # Helps in moving between various screens/windows
    widget.addWidget(initial_screen)
    widget.setFixedHeight(801) # Fixing the Height of the GUI Window to 800
    widget.setFixedWidth(1201) # Fixing the Width of the GUI Window to 1200
    widget.setWindowTitle("Data processing") # Setting Window Title
    widget.show() # Displaying the whole Application
    try: # In case the app doesn't exit.
        sys.exit(app.exec())
    except:
        print("Exiting...") # Printing confirmation message of Application exit in VS Terminal.