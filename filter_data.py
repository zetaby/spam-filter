import numpy as np
import re
import email
import base64
import codecs
from bs4 import BeautifulSoup
from stemming.porter2 import stem
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')


def setOfWordsToVecTor(vocabularyList, emailWords):

    vocabMarked = [0] * len(vocabularyList)
    for word in emailWords:
        if word in vocabularyList:
            vocabMarked[vocabularyList.index(word)] += 1
    return vocabMarked

def classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testWords):

    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)

    # ci = argmax (log pi + sum (log P(wj|ci)))
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)

    if p1 > p0:
        return 1
    else:
        return 0

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
    fr = open('pSpam.txt')
    pSpam = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam

def getVocabularyList(fileName):
    """
    get vovabularyList
    :param fileName:
    :return:
    """
    fr = open(fileName, 'r')
    vocabularyList=[]
    line = fr.readline()
    while line:
        vocabularyList.append(line.strip())
        line = fr.readline()
    fr.close()
    return vocabularyList

def loadEmail(filePath):
    inFile = open(filePath, 'r')
    content = inFile.read()
    content = content.decode('utf-8','ignore').encode('utf-8')
    regex = re.compile(r'[^a-zA-Z]|\d')
    words = regex.split(_getSubjectBodyText(content))
    filteredwords = [stem(w) for w in words if w and w not in stopwords]
    return filteredwords

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

def _getTextFromDifferentContentType(contentType,contentEncoding, text):
    if contentType == 'text/plain':
        if contentEncoding == 'base64':
            text = base64.standard_b64decode(text)
        return str(text)
    elif contentType == 'text/html':
        if contentEncoding == 'base64':
            text = base64.standard_b64decode(text)

        soup = BeautifulSoup(text, 'lxml')
        clean_html = ' '.join(soup.findAll(text=True))
        return clean_html
    else:
        return ''

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
             "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",
             "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",
             "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind",
             "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can",
             "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due",
             "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever",
             "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first",
             "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go",
             "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
             "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its",
             "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might",
             "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither",
             "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",
             "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out",
             "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems",
             "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
             "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their",
             "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
             "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
             "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
             "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
             "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever",
             "whole", "whom", "whose", "why",  "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

