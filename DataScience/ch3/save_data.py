import requests, zipfile
import io
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'

r = requests.get(url, stream=True)

# zipfileをurlから取得する
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
