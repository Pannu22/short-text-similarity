#%%
import itertools

# %%
from collections import Counter

# %%
import keras
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

# %%
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

#%%
from keras import metrics, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    concatenate,
)
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# %%
stop_words = nltk.corpus.stopwords.words("english")
lemma = nltk.stem.WordNetLemmatizer()

# %%
def preprocess_document(corpus):
    """Function to preprocess the corpus. Following actions will be performed :-
    - words will be converted to lower case
    - redundant spaces will be removed
    - stopwords from nltk library will be removed 

    Arguments:
        corpus {[String]} -- [Sentences]

    Returns:
        [String] -- [Cleaned sentence]
    """
    # lower the string and strip spaces
    corpus = corpus.lower()
    corpus = corpus.strip()

    # tokenize the words in document
    word_tokens = nltk.WordPunctTokenizer().tokenize(corpus)

    # remove stopwords
    filtered_tokens = [token for token in word_tokens if token not in stop_words]

    # lemmatize the word
    lemmatized_tokens = [lemma.lemmatize(token) for token in filtered_tokens]

    # join document from the tokens
    corpus = " ".join(lemmatized_tokens)

    return corpus


#%%
# Loading the data
data = pd.read_csv("../input/train.csv")

data.head()

data["label"].value_counts() / data["label"].value_counts().sum()

#%%
# Vectorizing function so that it can work on corpus
preprocess_document = np.vectorize(preprocess_document)

# %%
data["sentence1_cleaned"] = preprocess_document(data["sentence1"])
data["sentence2_cleaned"] = preprocess_document(data["sentence2"])

token_data = pd.concat([data["sentence1_cleaned"], data["sentence2_cleaned"]])


#%%
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(token_data)
#%%
sentence1_sequences = tok.texts_to_sequences(data["sentence1_cleaned"])
sentence2_sequences = tok.texts_to_sequences(data["sentence2_cleaned"])
sentence1_matrix = sequence.pad_sequences(sentence1_sequences, maxlen=max_len)
sentence2_matrix = sequence.pad_sequences(sentence2_sequences, maxlen=max_len)

#%%
sentence1 = Input(shape=[max_len], name="sentence1")
sentence1_embedding = Embedding(max_words, 50, input_length=max_len)(sentence1)

sentence2 = Input(shape=[max_len], name="sentence2")
sentence2_embedding = Embedding(max_words, 50, input_length=max_len)(sentence2)

concat = concatenate([sentence1_embedding, sentence2_embedding])


hidden1 = Dense(64, activation="relu")(concat)
batch_norm_hidden_1 = BatchNormalization()(hidden1)
hidden2 = Dense(64, activation="relu")(hidden1)
flatten = Flatten()(hidden2)
output = Dense(1, name="output", activation="sigmoid")(flatten)

#%%
model = keras.Model(inputs=[sentence1, sentence2], output=[output])
#%%
model.summary()
#%%
# Creating an ADAM optimizer
adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)


# %%
# compiling model
model.compile(
    optimizer=adam,
    loss="binary_crossentropy",
    metrics=[metrics.Precision(), metrics.Recall()],
)

#%%
model.fit(
    x=[sentence1_matrix, sentence2_matrix],
    y=y.values,
    batch_size=64,
    epochs=100,
    verbose=1,
    validation_split=0.25,
)

#%%
test_data = pd.read_csv("../input/test.csv")
test_data.head()
#%%
test_data["sentence1_cleaned"] = preprocess_document(test_data["sentence1"])
test_data["sentence2_cleaned"] = preprocess_document(test_data["sentence2"])

#%%
test_sentence1_sequences = tok.texts_to_sequences(test_data["sentence1_cleaned"])
test_sentence2_sequences = tok.texts_to_sequences(test_data["sentence2_cleaned"])
test_sentence1_matrix = sequence.pad_sequences(test_sentence1_sequences, maxlen=max_len)
test_sentence2_matrix = sequence.pad_sequences(test_sentence2_sequences, maxlen=max_len)
#%%
test_data["label"] = model.predict(x=[test_sentence1_matrix, test_sentence2_matrix])

#%%
test_data["label"] = test_data["label"] >= 0.5


#%%
test_data["label"] = test_data["label"] * 1

#%%
test_data.head()
test_data[["pid", "label"]].to_csv("../output/submission_1.csv", index=False)
#%%
# data[["sentence1_cleaned", "sentence2_cleaned", "label"]].to_csv(
#     "../output/sample_training_data.csv", index=True
# )


# # %%
# glove_input_file = "../model/glove.twitter.27B/glove.twitter.27B.25d.txt"
# word2vec_output_file = "../model/word2vec.txt"
# glove2word2vec(glove_input_file, word2vec_output_file)

# model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


# #%%
# # Check if word2vec is loaded or not
# print(
#     model.most_similar(positive=["australia"], topn=10)
# )  # [('spain', 0.8764086365699768), ('london', 0.8526231646537781), ('uk', 0.8515673875808716), ('germany', 0.8489168882369995), ('canada', 0.8392830491065979), ('denmark', 0.8238717317581177), ('singapore', 0.8228336572647095), ('vietnam', 0.8192894458770752), ('europe', 0.8162673711776733), ('asia', 0.8147203922271729)]
# print(model.similarity("woman", "man"))  # 0.70595723

# # %%
# data.head()

# # %%
# data["sentence1_cleaned"][0]

# # %%
# def calcWmDistance(sentence1, sentence2):
#     wmdistance = model.wmdistance(sentence1, sentence2)
#     return wmdistance


# vect_calcWmDistance = np.vectorize(calcWmDistance)


# # %%
# data["wmdistance"] = vect_calcWmDistance(data["sentence1"], data["sentence2"])
# # %%
# data.head()

# # %%
# sns.scatterplot(x="label", y="wmdistance", data=data)


# %%
word_emb_model = KeyedVectors.load_word2vec_format("../model/wiki-news-300d-1M.vec")


def map_word_frequency(document):
    return Counter(itertools.chain(*document))


def get_sif_feature_vectors(sentence1, sentence2, word_emb_model=word_emb_model):
    sentence1 = [
        token for token in sentence1.split() if token in word_emb_model.wv.vocab
    ]
    sentence2 = [
        token for token in sentence2.split() if token in word_emb_model.wv.vocab
    ]
    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 300  # size of vectore in word embeddings
    a = 0.001
    sentence_set = []
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
            vs = np.add(
                vs, np.multiply(a_value, word_emb_model.wv[word])
            )  # vs += sif * word_vector
        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)
    return sentence_set


# %%
sif = get_sif_feature_vectors(
    data["sentence1_cleaned"][0], data["sentence2_cleaned"][0]
)


# %%
data["sentence1_vector"] = ""
data["sentence2_vector"] = ""


# %%
for i in range(data.shape[0]):
    data["sentence1_vector"][i], data["sentence2_vector"][i] = get_sif_feature_vectors(
        data["sentence1_cleaned"][i], data["sentence2_cleaned"][i]
    )
# %%
data.head()
# %%
X = data[["sentence1_vector", "sentence2_vector"]]
y = data["label"].values
merged_array = np.stack(
    [data["sentence1_vector"].values, data["sentence2_vector"].values], axis=1
)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    merged_array, y, test_size=0.2, random_state=3
)
#%%
INPUT_LAYER = X_train.shape[1]

#%%
# Instantiate keras sequential model

# %%
sentence1_vector = X["sentence1_vector"]
sentence2_vector = X["sentence2_vector"]
y = data["label"]

#%%
# fitting the model on training set
model.fit(
    x=[sentence1_vector.values, sentence2_vector.values],
    y=y.values,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_split=0.25,
)

# %%
# evaluating on unknown data
score = model.evaluate(x=X_test, y=y_test)
