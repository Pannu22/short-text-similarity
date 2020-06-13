#%%
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

# %%
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

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

#%%
# data[["sentence1_cleaned", "sentence2_cleaned", "label"]].to_csv(
#     "../output/sample_training_data.csv", index=True
# )


# %%
glove_input_file = "../model/glove.twitter.27B/glove.twitter.27B.25d.txt"
word2vec_output_file = "../model/word2vec.txt"
glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


#%%
# Check if word2vec is loaded or not
print(
    model.most_similar(positive=["australia"], topn=10)
)  # [('spain', 0.8764086365699768), ('london', 0.8526231646537781), ('uk', 0.8515673875808716), ('germany', 0.8489168882369995), ('canada', 0.8392830491065979), ('denmark', 0.8238717317581177), ('singapore', 0.8228336572647095), ('vietnam', 0.8192894458770752), ('europe', 0.8162673711776733), ('asia', 0.8147203922271729)]
print(model.similarity("woman", "man"))  # 0.70595723

# %%
data.head()

# %%
data["sentence1_cleaned"][0]

# %%
def calcWmDistance(sentence1, sentence2):
    wmdistance = model.wmdistance(sentence1, sentence2)
    return wmdistance


vect_calcWmDistance = np.vectorize(calcWmDistance)


# %%
data["wmdistance"] = vect_calcWmDistance(data["sentence1"], data["sentence2"])
# %%
data.head()

# %%
sns.scatterplot(x="label", y="wmdistance", data=data)


# %%
word_emb_model = KeyedVectors.load_word2vec_format("../model/wiki-news-300d-1M.vec")


# %%
from collections import Counter
import itertools


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
data["sentence1_vector"], data["sentence2_vector"] = np.vectorize
