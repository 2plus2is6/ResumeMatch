import os
import string

cv_storage = {}


for file in os.listdir("input_cvs/"):
    if file.endswith(".txt"):
        with open(os.path.join("input_cvs",file), encoding='utf-8') as fName:
            text = fName.read()
            text = text.lower()
            cv_storage[file] = text


#print(cv_storage)

def keyWordExtraction():
    keywordInput = input("Please enter keyword all keywords separated by a comma: ")
    keywordInput = keywordInput.lower()
    keywordInput = [kw.strip() for kw in keywordInput.split(",")]
    return keywordInput


# print(keywordInput[2])

matchScore = {}

def keywordMatchesInAFile(fileName, fileData, keyWords):
    count = 0;
    for keyword in keyWords:
        if keyword in fileData:
            count+=1
    return count

listOfKeywords = keyWordExtraction()

for key,value in cv_storage.items():
    count = keywordMatchesInAFile(key,value, listOfKeywords)
    matchScore[key] = count

matchScore = sorted(matchScore.items(), key=lambda x: x[1], reverse=True)

print(matchScore)
