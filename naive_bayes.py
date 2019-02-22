from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes


conf = SparkConf().setAppName("Naive_Bayes")
sc   = SparkContext(conf=conf)

#word to vector space converter, limit to 10000 words
htf = HashingTF(10000)
#let 1 - positive class, 0 - negative class
#tokenize sentences and transform them into vector space model
positiveData = sc.textFile("rt-polarity-pos.txt")#TODO: load positive file
posdata = positiveData.map(lambda text : LabeledPoint(1, htf.transform(text.split(" "))))
print "No. of Positive Sentences: " + str(posdata.count())
posdata.persist()
negativeData = sc.textFile("rt-polarity-neg.txt")#TODO: load positive file
negdata = negativeData.map(lambda text : LabeledPoint(0, htf.transform(text.split(" "))))
print "No. of Negative Sentences: " + str(negdata.count())
negdata.persist()



# Split positive and negative data 60/40 into training and test data sets
ptrain, ptest = posdata.randomSplit([0.6, 0.4])
ntrain, ntest = negdata.randomSplit([0.6, 0.4])

#union train data with positive and negative sentences
trainh = ptrain.union(ntrain)
#union test data with positive and negative sentences
testh = ptest.union(ntest)

# Train a Naive Bayes model on the training data
model = NaiveBayes.train(trainh)

# Compare predicted labels to actual labels
# TODO here
prediction_and_labels = testh.map(lambda point: (model.predict(point.features), point.label))

# Filter to only correct predictions
# TODO here
correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

# Calculate and print accuracy rate
accuracy = correct.count() / float(testh.count())

print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"
