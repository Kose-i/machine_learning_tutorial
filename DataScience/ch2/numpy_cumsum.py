import numpy

data = numpy.array(range(0,10))
print(data)

print("dataの最小:{0}".format(data.min()))
print("dataの最大:{0}".format(data.max()))
print("dataの合計:{0}".format(data.sum()))
print("dataの平均:{0}".format(data.mean()))
print("dataの積上:{0}".format(data.cumsum()))
print("dataの積上割合:{0}".format(data.cumsum()/data.sum()))
