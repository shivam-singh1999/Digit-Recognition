from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc

digit = datasets.load_digits()
#print (digit)
features = digit.data
value = digit.target

#print(value,features)
#print (len(digit.data))

model = SVC(gamma=0.01)
model.fit(features,value)

#print(model.predict([feature]))

img = misc.imread("C:/Users/shivam singh/Desktop/ayla.jpg")
img = misc.imresize(img,(8,8))
img = img.astype(digit.images.dtype)

#print (img)

x_test = []

for eachrow in img:
    for eachpixel in eachrow:
        x_test.append(sum(eachpixel)/3.0)

print (model.predict([x_test]))