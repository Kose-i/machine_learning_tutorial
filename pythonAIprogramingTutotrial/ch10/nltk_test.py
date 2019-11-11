from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

input_text = "Do you know how tokenization works?\
              It's actually quite intersting!\
              Let's analyze a couple of sentences and figure it out."
print("Sentence tokenizer:")
print(sent_tokenize(input_text))

print("Word tokenizer:")
print(word_tokenize(input_text))

print("Word punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

stemmer_names = ['INPUT WORD', 'PORTER', 'LANCASTER', 'SNOWBALL']
fmt = '{:>16}'*len(stemmer_names)
print(fmt.format(*stemmer_names))
print('='*68)

for word in input_words:
    output = [word, porter.stem(word), lancaster.stem(word), snowball.stem(word)]
    print(fmt.format(*output))

from nltk.stem import WordNetLemmatizer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

lemmatizer = WordNetLemmatizer()
lemmatizer_names = ['INPUT WORD', 'NOUN LEMMATIZER', 'VERB LEMMATIZER']
fmt = '{:>24}'*len(lemmatizer_names)
print(fmt.format(*lemmatizer_names))
print('='*75)

for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'), lemmatizer.lemmatize(word, pos='v')]
    print(fmt.format(*output))

def chunker(input_data, N):
    input_words = input_data.split(' ')
    output = []
    while len(input_words) > N:
        output.append(' '.join(input_words[:N]))
        input_words = input_words[N:]
    output.append(' '.join(input_words))
    return output

from nltk.corpus import brown
input_data = ' '.join(brown.words()[:12000])
chunk_size = 700
chunks = chunker(input_data, chunk_size)
print('Number of text chunks =', len(chunks), '\n')
for i, chunk in enumerate(chunks):
    print('Chunk', i+1, '==>', chunk[:50])
