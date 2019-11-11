from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line.strip())
    return data
def process(input_text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input_text.lower())
    stop_words = stopwords.words('english')
    tokens = [x for x in tokens if not x in stop_words]
    stemmer = SnowballStemmer('english')
    tokens_stemmed = [stemmer.stem(x) for x in tokens]
    return tokens_stemmed

data = load_data('datasets/data.txt')
tokens = [process(x) for x in data]

dict_tokens = corpora.Dictionary(tokens)
doc_term_mat = [dict_tokens.doc2bow(token) for token in tokens]

num_topics = 2
ldamodel = models.ldamodel.LdaModel(doc_term_mat, num_topics=num_topics, id2word=dict_tokens, passes=25)

num_words = 5
print('Top ' + str(num_words) + ' contributing words to each topic:')
for n,values in ldamodel.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
    print('\nTopic', n)
    for word, weight in values:
        print(word, '==>', str(round(float(weight)*100, 2)) + '%')
