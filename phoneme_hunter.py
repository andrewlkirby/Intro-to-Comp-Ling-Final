import nltk
from nltk.tokenize import RegexpTokenizer
#import os
#os.chdir(r"C:\Users\Bertold\Documents\CUNY\Fall 2019\Intro to Computational Linguistics\Final") 

with open("DC_transcript.txt") as fin: 
    transcript = fin.read()
#Two text files are included. Paste above or just change DC to LB. Filenames:
    #LB_transcript.txt    (Lewis Black)
    #DC_transcript.txt    (Dave Chappelle)


regxptokenizer = RegexpTokenizer(r'\w+')

lowercasetext = transcript.lower()
nopuncttxt = regxptokenizer.tokenize(lowercasetext) 

arpabet = nltk.corpus.cmudict.dict()

def phoneme_counter(str):
    Kcount = 0    
    for word in nopuncttxt:
        try:
            print(arpabet[word][0])
        except KeyError:
            pass
        try:
            for j in range(len(arpabet[word][0])):
                try:
                    if arpabet[word][0][j] == "K":
                        Kcount += 1
                except KeyError:
                    pass
        except: 
            pass
    # UW1 = /u/ phoneme, like in 'boot'
    UW1count = 0    
    for word in nopuncttxt:
        try:
            for j in range(len(arpabet[word][0])):
                try:
                    if arpabet[word][0][j] == "UW1":
                        UW1count += 1
                except KeyError:
                    pass
        except: 
            pass
    print("The number of plosive /k/ sounds in this corpus is:", Kcount)
    print("The number of /u/ sounds in this corpus is:", UW1count)
    print("The number /k/ and /u/ sounds in this corpus is:", Kcount + UW1count)       

phoneme_counter(str)
    
