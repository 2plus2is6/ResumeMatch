import os

cv_storage = {}

for file in os.listdir("input_cvs/"):
    if file.endswith(".txt"):
        with open(os.path.join("input_cvs",file), encoding='utf-8') as fName:
            text = fName.read()
            text = text.lower()
            cv_storage[file] = text


#print(cv_storage)

keywordInput = input("Please enter keyword all keywords seperated by a comma: ")
keywordInput = keywordInput.replace(" ","")
keywordInput = keywordInput.split(",")

#print(keywordInput)