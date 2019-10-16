"""
python code:Learning and Predict for Iris Dataset
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Get Iris data
iris = load_iris()
x, y = iris.data, iris.target
# Split training_data, test_data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
# Initialize
nn = MLPClassifier(solver="sgd",random_state=0, max_iter=10000)
# Learning
nn.fit(x_train, y_train)
# Display predict-result
print("result:",nn.score(x_test, y_test))
