def chunker(input_data, N):
    input_words = input_data.split(' ')
    output = []
    while len(input_words) > N:
        output.append(' '.join(input_words[:N]))
        input_words = input_words[N:]
    output.append(' '.join(input_words))
    return output

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown

input_data = ' '.join(brown.words()[:5400])
chunk_size = 800
text_chunks = chunker(input_data, chunk_size)

count_vectorizer = CountVectorizer(min_df=7, max_df=20)
document_term_matrix = count_vectorizer.fit_transform(text_chunks)

vocabulary = count_vectorizer.get_feature_names()
print("Vocabulary:\n", vocabulary)

print("Document term matrix:")
fmt = '{:>8} '
for v in vocabulary:
    fmt += '{{:>{}}} '.format(len(v))
print(fmt.format('Document', *vocabulary))
for i, item in enumerate(document_term_matrix.toarray()):
    print(fmt.format('Chunk-' + str(i+1), *item.data))

