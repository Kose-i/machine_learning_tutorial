from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

def extract_features(words):
    return dict([(word, True) for word in words])

fileids_pos = movie_reviews.fileids('pos')
fileids_neg = movie_reviews.fileids('neg')

features_pos = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in fileids_pos]
features_neg = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in fileids_neg]

threshold = 0.8
num_pos = int(threshold * len(features_pos))
num_neg = int(threshold * len(features_neg))

features_train = features_pos[:num_pos] + features_neg[:num_neg]
features_test  = features_pos[num_pos:] + features_neg[num_neg:]

print('Number of training datapoints:', len(features_train))
print('Number of test     datapoints:', len(features_test))

classifier = NaiveBayesClassifier.train(features_train)
print('Accuracy of the classifier:', nltk_accuracy(classifier, features_test))

N = 15
print('Top ' + str(N) + ' most informative words:')
for i, item in enumerate(classifier.most_informative_features()[:N]):
    print(str(i+1) + '. ' + item[0])

input_reviews = [
    'The costumes in this movie were great',
    'I think the story was terrible and the characters were very weak',
    'People say that the director of the movie is amazing',
    'This is such an idiotic movie. I will not recommend it to anyone.'
]

print("Movie review predictions:")
for review in input_reviews:
    print("\nReview:", review)
    features = extract_features(review.split())
    probabilities = classifier.prob_classify(features)
    predicted_sentiment = probabilities.max()
    print("Predicted sentiment:", predicted_sentiment)
    print("Probability:", round(probabilities.prob(predicted_sentiment), 2))
