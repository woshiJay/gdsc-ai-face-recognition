import face_recognition
import pickle
import cv2
import os

folderPathImages = 'Images'
listPathImages = os.listdir(folderPathImages)
imgListImages= []

studentIDs = []
print(listPathImages)

for path in listPathImages: 
    imgListImages.append(cv2.imread(os.path.join(folderPathImages,path)))

    # taking only the first element after splitting text and store that into student IDs
    studentIDs.append(os.path.splitext(path)[0])

def generateEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodingsListKnow = generateEncodings(imgListImages)
encodingsListWithIDs = [encodingsListKnow, studentIDs]

# Open new file called EncodingsFile.p in WRITE MODE
encodingFile = open("EncodingsFile.p", "wb")
#Using pickle to dump into the file your encodings list
pickle.dump(encodingsListWithIDs, encodingFile)
#Close file
encodingFile.close()

print(studentIDs)