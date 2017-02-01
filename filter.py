import sys
import filter_data as filterData

filePath = sys.argv[1]
vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = \
    filterData.getTrainedModelInfo()

words = filterData.loadEmail(filePath)

mailType = \
    filterData.classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, words)

if mailType == 1:
    print("spam")
else:
    print("ham")