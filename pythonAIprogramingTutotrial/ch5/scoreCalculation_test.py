import json
import numpy as np

# Euclid score
def euclidean_score(dataset, usr1, usr2):
    if usr1 not in dataset:
        raise TypeError('Cannot find '+ usr1 +' in the dataset')
    if usr2 not in dataset:
        raise TypeError('Cannot find '+ usr2 +' in the dataset')
    common_movies = {}
    for item in dataset[usr1]:
        if item in dataset[usr2]:
            common_movies[item] = 1
    if len(common_movies) == 0:
        return 0
    squared_diff = []
    for item in common_movies:
        squared_diff.append(np.square(dataset[usr1][item] - dataset[usr2][item]))
    return 1 / (1+np.sqrt(np.sum(squared_diff)))
def pearson_score(dataset, usr1, usr2):
    if usr1 not in dataset:
        raise TypeError('Cannot find '+ usr1 +' in the dataset')
    if usr2 not in dataset:
        raise TypeError('Cannot find '+ usr2 +' in the dataset')
    common_movies = {}
    for item in dataset[usr1]:
        if item in dataset[usr2]:
            common_movies[item] = 1
    num_ratings = len(common_movies)
    if num_ratings == 0:
        return 0
    usr1_sum = np.sum([dataset[usr1][item] for item in common_movies])
    usr2_sum = np.sum([dataset[usr2][item] for item in common_movies])
    usr1_squared_sum = np.sum([np.square(dataset[usr1][item]) for item in common_movies])
    usr2_squared_sum = np.sum([np.square(dataset[usr2][item]) for item in common_movies])
    sum_of_products = np.sum([dataset[usr1][item]*dataset[usr2][item] for item in common_movies])
    Sxy = sum_of_products - (usr1_sum*usr2_sum / num_ratings )
    Sxx = usr1_squared_sum - np.square(usr1_sum) / num_ratings
    Syy = usr2_squared_sum - np.square(usr2_sum) / num_ratings
    if Sxx*Syy == 0:
        return 0
    return Sxy / np.sqrt(Sxx*Syy)
rating_file = 'datasets/ratings.json'

with open(rating_file, 'r') as f:
    data = json.loads(f.read())
usr1 = 'David Smith'
usr2 = 'Bill Duffy'

print("Euclidean score:")
print(euclidean_score(data, usr1, usr2))

print("Pearson score:")
print(pearson_score(data, usr1, usr2))

## Collaborative filtering
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find '+ user +'in the dataset')
    scores = [(x, pearson_score(dataset, user, x)) for x in dataset if x != user]
    scores.sort(key=lambda p: p[1], reverse=True)
    return scores[:num_users]
user = 'Bill Duffy'

print('Users similar to '+user+':\n')
similar_users = find_similar_users(data, user, 3)
print('User\t\tSimilarity score')
print('-'*41)
for item in similar_users:
    print(item[0], '\t\t', round(item[1], 2))

## Movie selection
def get_recommendations(dataset, input_user):
    similar_users = find_similar_users(dataset, input_user, 3)
    overall_scores = {}
    similarity_scores = {}
    for user, pscore in similar_users:
        for item,iscore in dataset[user].items():
            if item in dataset[input_user] and dataset[input_user][item] > 0:
                continue
            overall_scores[item] = overall_scores.get(item, 0) + iscore*pscore
            similarity_scores[item] = similarity_scores.get(item, 0) + pscore
    if len(overall_scores) == 0:
        return ['No recommendations possible']
    movie_scores = [(item, score / similarity_scores[item]) for item,score in overall_scores.items()]
    movie_scores.sort(key=lambda p: p[1], reverse=True)
    return movie_scores
user = 'Chris Duncan'

print("Movie recommendations for "+ user +":")
movies = get_recommendations(data, user)
for i, movie in enumerate(movies):
    print(str(i+1) + '.', movie[0], ':', round(movie[1], 2))
