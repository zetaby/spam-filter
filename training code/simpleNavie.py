import numpy as np
import os
import re
import email
import base64
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk import word_tokenize, WordNetLemmatizer
from stemming.porter2 import stem
from scipy.special import factorial
import nltk

stoplist = stopwords.words('english')
Wfo_Lambda = 0.5
Wfo_threshold = 0.01


def getFeature(emailText):
    """
    preprocess and get feature
    word_tokenize: Splitting the text by white spaces and punctuation marks
    lemmatizers: identify different forms of the same word only once
    (e.g. price and prices -> price)
    Word Stemming (removes suffix’s using Porters algorithm)
    stopwords: remove those words which do not contain important significance
    :param text: original email content
    :return: filtered email content
    """

    lemmatizer = WordNetLemmatizer()
    regex = re.compile(r'[^a-zA-Z]|\d')
    words = regex.split(_getSubjectBodyText(emailText))
    filteredwords = [stem(w) for w in words if w and not w.lower() in stoplist]
    #filteredwords = [w for w in words if w and not w.lower() in stoplist]
    #print(filteredwords)
    return filteredwords

# def loadMailDataTest(folderAddress):
#     classCategory = []  # 1 for spam，0 for ham
#     emailWords = []
#     file_list = os.listdir(folderAddress)
#     for rFile in file_list:
#         if rFile.find('ham') != -1:
#             classCategory.append(0)
#         elif rFile.find('spam') != -1:
#             classCategory.append(1)
#
#         filename = folderAddress + rFile
#         inFile = open(filename, 'r', errors='ignore')
#         content = inFile.read()
#         regex = re.compile(r'[^a-zA-Z]|\d')
#         lemmatizer = WordNetLemmatizer()
#         features = regex.split(_getSubjectBodyTextTest(content))
#         filteredwords = [stem(w) for w in features if w and not w.lower() in stoplist]
#         if filteredwords:
#             emailWords.append(filteredwords)
#         else:
#             print(rFile, "feature is empty")
#
#     return emailWords, classCategory

def loadMailData(folderAddress):
    """
    read Mails in specific folder
    :param folderAddress: the path of mail folder
    :return:
    """
    classCategory = []  # 1 for spam，0 for ham
    emailWords = []
    file_list = os.listdir(folderAddress)
    for rFile in file_list:
        if rFile.find('ham') != -1:
            classCategory.append(0)
        elif rFile.find('spam') != -1:
            classCategory.append(1)

        filename = folderAddress + rFile
        inFile = open(filename,'r', errors='ignore')
        content = inFile.read()
        # for line in inFile.readlines():
        #     lines += line
        features = getFeature(content)
        if features:
            emailWords.append(features)
        else:
            print(rFile,"feature is empty")


    return emailWords, classCategory

def createVocabularyList(smsWords):
    # :set[] erase repetition
    # """
    vocabularySet = set([])
    for words in smsWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList


def getVocabularyList(fileName):
    """
    get vovabularyList
    :param fileName:
    :return:
    """
    print("getVocabularyList")
    fr = open(fileName, 'r')
    vocabularyList=[]
    line = fr.readline()
    while line:
        vocabularyList.append(line.strip())
        line = fr.readline()
    fr.close()
    print("length of vocabularyList", len(vocabularyList))
    return vocabularyList


def setOfWordsToVecTor(vocabularyList, emailWords):

    vocabMarked = [0] * len(vocabularyList)
    for word in emailWords:
        if word in vocabularyList:
            vocabMarked[vocabularyList.index(word)] += 1
    return vocabMarked


def setOfWordsListToVecTor(vocabularyList, smsWordsList):
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList

def wfoFilter(preVocabularyList, emailWordsList, classLables):
    filteredVocabularyList = []
    vocabMarkedList = setOfWordsListToVecTor(preVocabularyList, emailWordsList)
    pVocabularyinSpam, pVocabularyinHam, pSpam = _calPvocabulary(vocabMarkedList, classLables)

    conditionTemp = pVocabularyinSpam / pVocabularyinHam
    wfo = np.zeros(len(vocabMarkedList[0]))

    for i in range(len(conditionTemp)):
        if conditionTemp[i] > 1:
            wfo[i] = pow(np.log(pVocabularyinSpam[i]) - np.log(pVocabularyinHam[i]), 1 - Wfo_Lambda) \
                     * pow(pVocabularyinSpam[i], Wfo_Lambda)
        else:
            wfo[i] = pow(np.log(pVocabularyinHam[i]) - np.log(pVocabularyinSpam[i]), 1 - Wfo_Lambda) \
                     * pow(pVocabularyinHam[i], Wfo_Lambda)

    for j in range(len(wfo)):
        if wfo[j] > Wfo_threshold:
            filteredVocabularyList.append(preVocabularyList[j])
    return filteredVocabularyList


def trainingNaiveBayes(trainMarkedWords, trainCategory):
    pVocabularyinSpam, pVocabularyinHam, pSpam = _calPvocabulary(trainMarkedWords, trainCategory)
    pWordsSpamicity = np.log(pVocabularyinSpam)
    pWordsHealthy = np.log(pVocabularyinHam)
    #pWordsSpamicity = pVocabularyinSpam
    #pWordsHealthy = pVocabularyinHam

    return pWordsSpamicity, pWordsHealthy, pSpam


def getTrainedModelInfo():
    """
    get trained model info including
    vocabularyList
    pWordsHealthy
    pWordsSpamicity
    pSpam
    """
    vocabularyList = getVocabularyList('vocabularyList.txt')
    pWordsHealthy = np.loadtxt('pWordsHealthy.txt', delimiter='\t')
    pWordsSpamicity = np.loadtxt('pWordsSpamicity.txt', delimiter='\t')
    DS = np.loadtxt('trainDS.txt', delimiter='\t')
    #print("length of pWordsHealthy: ", len(pWordsHealthy))
   # print("length of pWordsSpamicity: ", len(pWordsSpamicity))
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS


def classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testWords):

    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    print(testWordsCount)
    testWordsMarkedArray = np.array(testWordsCount)

    # ci = argmax (log pi + sum (log P(wj|ci)))
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)

    # np.power(pWordsSpamicity, testWordsMarkedArray) / factorial(testWordsMarkedArray, exact=False)

    if p1 > p0:
        return 1
    else:
        return 0

def classifyMSE(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testWords):

    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)

    # ci = argmax (log pi + sum (log P(wj|ci)))
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)

    # np.power(pWordsSpamicity, testWordsMarkedArray) / factorial(testWordsMarkedArray, exact=False)
    print("p1->", p1, " p0->", p0)
    return p1, p0
    # if p1 > p0:
    #     return 1
    # else:
    #     return 0

def adaboostClassify(vocabularyList, pWordsSpamicity, pWordsHealthy, DS, pSpam, testWordsMarkedArray):

    # ci = argmax (log pi + sum (log P(wj|ci)))* DS
    p1 = sum(testWordsMarkedArray * pWordsSpamicity ) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)

    if p1 > p0:
        return p1, p0, 1
    else:
        return p1, p0, 0

def adaboostClassifyForPredict(vocabularyList, pWordsSpamicity, pWordsHealthy, DS, pSpam, testWords):
    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)
    ps, ph, mailType = adaboostClassify(vocabularyList, pWordsSpamicity, pWordsHealthy, DS, pSpam, testWordsMarkedArray)

    return mailType

def adaboostPredict(testEmailWords):
    predicted = []
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS = getTrainedModelInfo()

    for i in range(len(testEmailWords)):
        emailType = adaboostClassifyForPredict(vocabularyList, pWordsSpamicity,
                             pWordsHealthy, DS, pSpam, testEmailWords[i])
        predicted.append(emailType)

    return predicted

def baselinePredict(testEmailWords):
    predicted = []
    for i in range(len(testEmailWords)):
        if len(testEmailWords[i]) > 200:
            predicted.append(0)
        else:
            predicted.append(1)
    return predicted

def predict(testEmailWords):
    """
    predict spam or ham depend on pretrained info
    :param testEmailWords:array including many email words need to predict
    :return: predicted array {1:spam (positive),0:ham (negative)}
    """
    predicted = []
    vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, DS = getTrainedModelInfo()

    for i in range(len(testEmailWords)):
        emailType = classify(vocabularyList, pWordsSpamicity,
                             pWordsHealthy, pSpam, testEmailWords[i])
        predicted.append(emailType)

    return predicted

def _getTextFromDifferentContentType(contentType,contentEncoding, text):
    if contentType == 'text/plain':
        if contentEncoding == 'base64':
            text = base64.standard_b64decode(text)
        return str(text)
    elif contentType == 'text/html':
        if contentEncoding == 'base64':
            text = base64.standard_b64decode(text)

        soup = BeautifulSoup(text, 'lxml')
        # for script in soup(["script", "style"]):
        #     script.extract()
        clean_html = ' '.join(soup.findAll(text=True))
        return clean_html
    else:
        return ''

def _getSubjectBodyText(emailText):
    email_message = email.message_from_string(emailText)
    subject = email_message.get('Subject')
    mailSubjectBody = []
    if subject != None:
        mailSubjectBody.append(subject)
    if email_message.is_multipart():
        for part in email_message.get_payload():
            if part.is_multipart():
                for subPart in part.get_payload():
                    mailSubjectBody.append(_getTextFromDifferentContentType
                                           (subPart.get_content_type(),
                                            subPart['Content-Transfer-Encoding'],subPart.get_payload()))
            else:
                mailSubjectBody.append(_getTextFromDifferentContentType
                                       (part.get_content_type(),
                                        part['Content-Transfer-Encoding'],part.get_payload()))
    else:
        mailSubjectBody.append(_getTextFromDifferentContentType
                               (email_message.get_content_type(),
                                email_message['Content-Transfer-Encoding'],email_message.get_payload()))


    return ' '.join(mailSubjectBody)

def _calPvocabulary(trainMarkedWords, trainCategory):
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])
    # P(Spam) - prior probability
    pSpam = sum(trainCategory) / float(numTrainDoc)

    # count the occurrence of each vocabularyList(corpus) word in spam and ham
    # Laplace correction
    wordsInSpamNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  # spam
            wordsInSpamNum += trainMarkedWords[i]
            spamWordsNum += sum(trainMarkedWords[i])  # count total number of words in spam
        else:
            wordsInHealthNum += trainMarkedWords[i]  # ham
            healthWordsNum += sum(trainMarkedWords[i])  # count total number of words in ham

    pVocabularyinSpam = wordsInSpamNum / spamWordsNum
    pVocabularyinHam = wordsInHealthNum / healthWordsNum

    return pVocabularyinSpam, pVocabularyinHam, pSpam

# def _getSubjectBodyTextTest(emailText):
#     email_message = email.message_from_string(emailText)
#     subject = email_message.get('Subject')
#     mailSubjectBody = []
#     if subject != None:
#         mailSubjectBody.append(subject)
#     if email_message.is_multipart():
#         for part in email_message.get_payload():
#             if part.is_multipart():
#                 for subPart in part.get_payload():
#                     mailSubjectBody.append(_getTextFromDifferentContentTypeTest
#                                            (subPart.get_content_type(),
#                                             subPart['Content-Transfer-Encoding'],subPart.get_payload()))
#             else:
#                 mailSubjectBody.append(_getTextFromDifferentContentTypeTest
#                                        (part.get_content_type(),
#                                         part['Content-Transfer-Encoding'],part.get_payload()))
#     else:
#         mailSubjectBody.append(_getTextFromDifferentContentTypeTest
#                                (email_message.get_content_type(),
#                                 email_message['Content-Transfer-Encoding'],email_message.get_payload()))
#
#
#     return ' '.join(mailSubjectBody)

# def _getTextFromDifferentContentTypeTest(contentType,contentEncoding, text):
#     if contentType == 'text/plain':
#         if contentEncoding == 'base64':
#             text = base64.standard_b64decode(text)
#         return str(text)
#     elif contentType == 'text/html':
#         if contentEncoding == 'base64':
#             text = base64.standard_b64decode(text)
#         # soup = BeautifulSoup(text, 'lxml')
#         # clean_html = ' '.join(soup.findAll(text=True))
#         return cleanhtml(str(text))
#     else:
#         return ''

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext