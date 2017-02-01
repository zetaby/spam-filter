import numpy as np
import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import simpleNavie as naiveBayes
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

k_fold_num = 10

def simpleTest():
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS = \
        naiveBayes.getTrainedModelInfo()

    fileFolder = './test/'
    smsWords, classLables = naiveBayes.loadMailData(fileFolder)

    smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                  pWordsHealthy, pSpam, smsWords[0])
    print(smsType)

def testClassifyErrorRate():
    fileFolder = './public/'
    smsWords, classLables = naiveBayes.loadMailDataTest(fileFolder)
    #smsWords, classLables = naiveBayes.loadMailData(fileFolder)

    testWords = smsWords
    testWordsType = classLables

    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS = \
        naiveBayes.getTrainedModelInfo()

    errorCount = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(classLables)):
        # smsType = naiveBayes.adaboostClassifyForPredict(vocabularyList, pWordsSpamicity,
        #                               pWordsHealthy, DS, pSpam, testWords[i])
        smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                                        pWordsHealthy, pSpam, testWords[i])
        #print('predicted:', smsType, ' actual:', testWordsType[i])
        if smsType != testWordsType[i]:
            print(i)
            if(smsType == 1):
                fp += 1
            else:
                fn += 1
        else:
            if (smsType == 1):
                tp += 1
            else:
                tn += 1
    print(
    """
    Predicted:    | SPAM | HAM
    ----------------------------
    Ground Truth: |      |
        SPAM      | %4d | %4d
        HAM       | %4d | %4d
    """ % (tp, fn, fp, tn))
    acc = 100.0 * (tp + tn) / (fp + fn + tp + tn)
    print ("acc->",acc)


def crossValidateEvaluate():
    beginTime = datetime.datetime.now()
    filename = './public/'
    # load data: load all the words in all the emails
    mailWords, classLables = naiveBayes.loadMailData(filename)

    skf = StratifiedKFold(classLables, k_fold_num)
    acc_per_fold = []
    f1_per_fold = []
    recall_per_fold = []
    precision_per_fold = []

    for train_index, test_index in skf:
        print("train_index->", train_index)
        print("test_index->", test_index)
        preVocabularyList = naiveBayes.createVocabularyList([mailWords[i] for i in train_index])
        #do wfo filter
        vocabularyList = naiveBayes.wfoFilter(preVocabularyList,
                                              [mailWords[i] for i in train_index],
                                              [classLables[i] for i in train_index])
        vocabularyList = preVocabularyList
        print("length of vocabularyList", len(vocabularyList))
        fw = open('vocabularyList.txt', 'w')
        for i in vocabularyList:
            fw.write(i + '\n')
        fw.flush()
        fw.close()
        print("vocabularyList finished")

        trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, [mailWords[i] for i in train_index])
        print("trainMarkedWords finished")

        # change it to array
        trainMarkedWords = np.array(trainMarkedWords)
        print("data to matrix finished")
        # calculate each propabilaty of spam and ham P(wi/s)  p(wi/h)
        pWordsSpamicity, pWordsHealthy, pSpam = \
            naiveBayes.trainingNaiveBayes(trainMarkedWords, [classLables[i] for i in train_index])
        fpSpam = open('pSpam.txt', 'w')
        spam = pSpam.__str__()
        fpSpam.write(spam)
        fpSpam.close()

        np.savetxt('pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')
        np.savetxt('pWordsHealthy.txt', pWordsHealthy, delimiter='\t')

        predict = naiveBayes.predict( [mailWords[i] for i in test_index])
        #predict = naiveBayes.adaboostPredict([smsWords[i] for i in test_index])
        acc_per_fold.append(accuracy_score([classLables[i] for i in test_index], predict))
        f1_per_fold.append(f1_score([classLables[i] for i in test_index], predict))
        recall_per_fold.append(recall_score([classLables[i] for i in test_index], predict))
        precision_per_fold.append(precision_score([classLables[i] for i in test_index], predict))
        print("acc_per_fold:", acc_per_fold)
        print("f1_per_fold:", f1_per_fold)
        print("recall_per_fold:", recall_per_fold)
        print("precision_per_fold:", precision_per_fold)

    print("acc_per_fold:", acc_per_fold)
    print("f1_per_fold:", f1_per_fold)
    print("recall_per_fold:", recall_per_fold)
    print("precision_per_fold:", precision_per_fold)
    print("k-fold:", k_fold_num, " spend:", (datetime.datetime.now() - beginTime))


def testClassifyErrorRateByIndex():
    fileFolder = './public/'
    smsWords, classLables = naiveBayes.loadMailDataTest(fileFolder)

    test_index = [2,6,7,8,13,16,19,29,35,37,40,42,43,45,46,49,51,52,64,65,71,72,78,79,80,84,85,
                  90,91,98,103,109,111,117,123,129,135,138,142,149,169,188,191,192,203,221,225,226,
                  229,232,236,243,250,254,257,258,259,264,268,281,298,300,308,319,322,329,333,335,338,339,340,344,
                  347,358,359,362,382,385,391,394,402,410,415,417,418,422,423,424,425,428,437,441,456,461,462,470,472,477,480,481]

    testWords = [smsWords[i] for i in test_index]
    testWordsType = [classLables[i] for i in test_index]

    # testCount = 200
    # for i in range(testCount):
    #     randomIndex = int(random.uniform(0, len(classLables)))
    #     testWordsType.append(classLables[randomIndex])
    #     testWords.append(smsWords[randomIndex])
    #     del (smsWords[randomIndex])
    #     del (classLables[randomIndex])

    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS = \
        naiveBayes.getTrainedModelInfo()

    errorCount = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(test_index)):
        smsType = naiveBayes.classify(vocabularyList, pWordsSpamicity,
                                      pWordsHealthy, pSpam, testWords[i])
        print('predicted:', smsType, ' actual:', testWordsType[i])
        if smsType != testWordsType[i]:
            if(smsType == 1):
                fp += 1
            else:
                fn += 1
        else:
            if (smsType == 1):
                tp += 1
            else:
                tn += 1
    print(
    """
    Predicted:    | SPAM | HAM
    ----------------------------
    Ground Truth: |      |
        SPAM      | %4d | %4d
        HAM       | %4d | %4d
    """ % (tp, fn, fp, tn))
    acc = (tp + tn) / (fp + fn + tp + tn)
    print ("acc->",acc)


def baselineCrossValidateEvaluate():
    beginTime = datetime.datetime.now()
    filename = './public/'
    # load data: load all the words in all the emails
    mailWords, classLables = naiveBayes.loadMailData(filename)

    skf = StratifiedKFold(classLables, k_fold_num)
    acc_per_fold = []

    for train_index, test_index in skf:
        print("train_index->", train_index)
        print("test_index->", test_index)

        predict = naiveBayes.baselinePredict([mailWords[i] for i in test_index])

        acc_per_fold.append(accuracy_score([classLables[i] for i in test_index], predict))
        print("acc_per_fold:", acc_per_fold)

    print("acc_per_fold:", acc_per_fold)
    print("avg acc:", np.mean(acc_per_fold))
    print("k-fold:", k_fold_num, " spend:", (datetime.datetime.now() - beginTime))

def testClassifyErrorRateMSE():
    fileFolder = './public/'
    mailWords, classLables = naiveBayes.loadMailData(fileFolder)

    test_index = [2,6,7,8,13,16,19,29,35,37,40,42,43,45,46,49,51,52,64,65,71,72,78,79,80,84,85,
                  90,91,98,103,109,111,117,123,129,135,138,142,149,169,188,191,192,203,221,225,226,
                  229,232,236,243,250,254,257,258,259,264,268,281,298,300,308,319,322,329,333,335,338,339,340,344,
                  347,358,359,362,382,385,391,394,402,410,415,417,418,422,423,424,425,428,437,441,456,461,462,470,472,477,480,481]

    testWords = [mailWords[i] for i in test_index]
    testWordsType = [classLables[i] for i in test_index]

    # testCount = 200
    # for i in range(testCount):
    #     randomIndex = int(random.uniform(0, len(classLables)))
    #     testWordsType.append(classLables[randomIndex])
    #     testWords.append(smsWords[randomIndex])
    #     del (smsWords[randomIndex])
    #     del (classLables[randomIndex])

    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS = \
        naiveBayes.getTrainedModelInfo()

    errorCount = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    se = 0
    for i in range(len(test_index)):
        testWordsCount = naiveBayes.setOfWordsToVecTor(vocabularyList, testWords[i])
        trainMarkedWords = np.array(testWordsCount)
        p1, p0, type = naiveBayes.adaboostClassify(vocabularyList, pWordsSpamicity,
                                      pWordsHealthy,DS, pSpam, trainMarkedWords)

        autual = testWordsType[i]
        if autual == 1:
            se += (pow((p1 / 20000 - 1), 2) + pow((p0 / 20000), 2)) / 2
        else:
            se += (pow((p1 / 20000 ), 2) + pow((p0 / 20000 - 1), 2)) / 2

    print("mse->", se / len(test_index))
#crossValidateEvaluate()
#crossValidateEvaluate()
#testClassifyErrorRateByIndex()
#testClassifyErrorRate()
#baselineCrossValidateEvaluate()
testClassifyErrorRate()


