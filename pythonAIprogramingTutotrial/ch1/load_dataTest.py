from sklearn import datasets
house_prices = datasets.load_boston()
print(house_prices.data)
print(house_prices.target)
digits = datasets.load_digits()
print(digits.images[4])
