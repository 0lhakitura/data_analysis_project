import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel

df = pd.read_csv('consumer_complaints.csv', on_bad_lines='skip')
corpus = df['Consumer complaint narrative']
stop_words_1 = stopwords.words('english')
stop_words_2 = ['xxxx', 'xxxxxxxxxxxx']
stop_words = stop_words_1 + stop_words_2


# Text Preprocessing
def preprocess(data):
    reviews = []
    for review in data:
        words = word_tokenize(str(review))
        sentence = [w.lower() for w in words if w.isalpha()]
        reviews.append(' '.join([word for word in sentence if word not in stop_words and word != 'nan']))
    return reviews


# Bag of Words
def bag_of_words(preprocessed_data):
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(preprocessed_data)
    return data
    # data = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names_out())
    # print(data)


# TF-IDF
def tf_idf(preprocessed_data):
    vectorizer = TfidfVectorizer(min_df=1)
    data = vectorizer.fit_transform(preprocessed_data)
    return data
    # data = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names_out())
    # print(data)


# Latent Sematic Analysis (LSA)

def latent_semantic_analysis(data):
    lsa_model = TruncatedSVD(n_components=2,
                             algorithm='randomized',
                             n_iter=10)
    lsa = lsa_model.fit_transform(data)
    index = 0
    for j in lsa:
        print("Review " + str(index) + ": ")
        for i, topic in enumerate(lsa[index]):
            print("Topic ", i, ": ", topic * 100, "%")
        index += 1


# Latent Dirichlet Allocation (LDA)
def latent_dirichlet_allocation(documents):
    dictionary = corpora.Dictionary([document.split() for document in documents])
    corpus = [dictionary.doc2bow(document.split()) for document in documents]
    lda = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
    topics = lda.print_topics()
    for topic in topics:
        print(topic)
    for i, document in enumerate(documents):
        print(f"Document {i + 1}: {document}")
        print(lda.get_document_topics(corpus[i]))


if __name__ == '__main__':
    print(preprocess(corpus))
    print('latent_dirichlet_allocation')
    latent_dirichlet_allocation(preprocess(corpus))
    print('latent_semantic_analysis + tf_idf')
    latent_semantic_analysis(tf_idf(preprocess(corpus)))
    print('latent_semantic_analysis + bag_of_words')
    latent_semantic_analysis(bag_of_words(preprocess(corpus)))

