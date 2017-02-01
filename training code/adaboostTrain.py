import random
import numpy as np
import datetime
import simpleNavie as naiveBayes

def trainingAdaboostGetDS(iterateNum = 40):
    test_index = [2, 6, 7, 8, 13, 16, 19, 29, 35, 37, 40, 42, 43, 45, 46, 49, 51, 52, 64, 65, 71, 72, 78, 79, 80, 84,
                  85,
                  90, 91, 98, 103, 109, 111, 117, 123, 129, 135, 138, 142, 149, 169, 188, 191, 192, 203, 221, 225, 226,
                  229, 232, 236, 243, 250, 254, 257, 258, 259, 264, 268, 281, 298, 300, 308, 319, 322, 329, 333, 335,
                  338, 339, 340, 344,
                  347, 358, 359, 362, 382, 385, 391, 394, 402, 410, 415, 417, 418, 422, 423, 424, 425, 428, 437, 441,
                  456, 461, 462, 470, 472, 477, 480, 481]
    beginTime = datetime.datetime.now()
    filename = './public/'
    # load data: load all the words in all the emails
    mailWords, classLables = naiveBayes.loadMailData(filename)

    preVocabularyList = naiveBayes.createVocabularyList(mailWords)
    # do wfo filter
    vocabularyList = naiveBayes.wfoFilter(preVocabularyList, mailWords, classLables)
    print("length of vocabularyList", len(vocabularyList))

    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, mailWords)
    print("trainMarkedWords finished")

    # change it to array
    trainMarkedWords = np.array(trainMarkedWords)
    print("data to matrix finished")
    # calculate each propabilaty of spam and ham P(wi/s)  p(wi/h)
    pWordsSpamicity, pWordsHealthy, pSpam = \
        naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    DS = np.ones(len(vocabularyList))

    ds_result = {}
    minErrorRate = np.inf
    for i in range(iterateNum):
        errorCount = 0.0
        for j in test_index:
            testWordsCount = naiveBayes.setOfWordsToVecTor(vocabularyList, mailWords[j])
            testWordsMarkedArray = np.array(testWordsCount)
            ps, ph, mailType = naiveBayes.adaboostClassify(vocabularyList, pWordsSpamicity,
                                        pWordsHealthy, DS, pSpam, testWordsMarkedArray)

            if mailType != classLables[j]:
                errorCount += 1
                alpha = ps - ph
                if alpha > 0:  # actual: ham; predict:spam
                    DS[testWordsMarkedArray != 0] = np.abs((DS[testWordsMarkedArray != 0] - np.exp(alpha)) / DS[testWordsMarkedArray != 0])
                else:  # actual: spam; predict: ham
                    DS[testWordsMarkedArray != 0] = (DS[testWordsMarkedArray != 0] + np.exp(alpha)) / DS[testWordsMarkedArray != 0]

        print('DS:', DS)
        errorRate = errorCount / len(mailWords)
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            ds_result['minErrorRate'] = minErrorRate
            ds_result['DS'] = DS
        print('# %d，errorcount %d ，errorrate %f' % (i, errorCount, errorRate))
        if errorRate == 0.0:
            break

    ds_result['vocabularyList'] = vocabularyList
    ds_result['pWordsSpamicity'] = pWordsSpamicity
    ds_result['pWordsHealthy'] = pWordsHealthy
    ds_result['pSpam'] = pSpam
    return ds_result

if __name__ == '__main__':
    dsErrorRate = trainingAdaboostGetDS()
    np.savetxt('pWordsSpamicity.txt', dsErrorRate['pWordsSpamicity'], delimiter='\t')
    np.savetxt('pWordsHealthy.txt', dsErrorRate['pWordsHealthy'], delimiter='\t')
    np.savetxt('pSpam.txt', np.array([dsErrorRate['pSpam']]), delimiter='\t')
    np.savetxt('trainDS.txt', dsErrorRate['DS'], delimiter='\t')
    np.savetxt('trainMinErrorRate.txt', np.array([dsErrorRate['minErrorRate']]), delimiter='\t')
    vocabulary = dsErrorRate['vocabularyList']
    fw = open('vocabularyList.txt', 'w')
    for i in range(len(vocabulary)):
        fw.write(vocabulary[i] + '\n')
    fw.flush()
    fw.close()