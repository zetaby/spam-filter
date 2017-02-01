import numpy as np
import datetime
import simpleNavie as naiveBayes

beginTime = datetime.datetime.now()
filename = './public/'
#load data: load all the words in all the emails
smsWords, classLables = naiveBayes.loadMailData(filename)

#get the non-repeated features(vacabulary)
preVocabularyList = naiveBayes.createVocabularyList(smsWords)
#do wfo filter
vocabularyList = naiveBayes.wfoFilter(preVocabularyList, smsWords, classLables)

print("length of vocabularyList", len(vocabularyList))
fw = open('vocabularyList.txt', 'w')
for i in vocabularyList:
    fw.write(i + '\n')
fw.flush()
fw.close()
print( "vocabularyList finished")

#change to vector: each email is a vector included in the trainMakredWords
trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
print( "trainMarkedWords finished")
# change it to array
trainMarkedWords = np.array(trainMarkedWords)
print(  "data to matrix finished")
#calculate each propabilaty of spam and ham P(wi/s)  p(wi/h)
pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)
print("length of pWordsSpamicity:", len(pWordsSpamicity))
print("length of pWordsHealthy:", len(pWordsHealthy))
print( 'pSpam:', pSpam)
fpSpam = open('pSpam.txt', 'w')
spam = pSpam.__str__()
fpSpam.write(spam)
fpSpam.close()

np.savetxt('pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')
np.savetxt('pWordsHealthy.txt', pWordsHealthy, delimiter='\t')
print( 'training finished, spend: ', (datetime.datetime.now() - beginTime) , " ms")