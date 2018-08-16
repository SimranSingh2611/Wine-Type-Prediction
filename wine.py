from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split as tts

wine = datasets.load_wine()

features = wine.data
labels = wine.target

print ('Number of entries:', len(features))
for featurename in wine.feature_names:
    print (featurename[:10],end="\t")
print ('Class')
for feature, label in zip(features, labels):
    for f in feature:
        print (f,end="\t\t"),
    print (label)


train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.2)
print ('Number of entries:', len(train_feats))
for featurename in wine.feature_names:
    print (featurename[:10],end="\t")
print ('Class')
for feature, label in zip(train_feats, train_labels):
    for f in feature:
        print (f,end="\t\t"),
    print (label)


clf=svm.SVC()
clf.fit(train_feats, train_labels)

predictions=clf.predict(test_feats)
print (predictions)

score=0
for i in range(0,len(predictions)):
    if predictions[i]==test_labels[i]:
        score+=1

print(score/len(predictions))


