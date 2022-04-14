import struct
import numpy as np

def getLabels(filePath):
    with open(filePath, 'rb') as file:
        magic, n = struct.unpack('>II',file.read(8))
        labels = np.fromfile(file,dtype=np.uint8)
        return labels

def getImgs(filePath):
    with open(filePath, 'rb') as file:
        magic, n, rows, colums = struct.unpack('>IIII',file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(n, rows, colums)
        return images

if __name__ == '__main__':
    trainLabels = getLabels('MNIST/train-labels-idx1-ubyte')
    testLabels = getLabels('MNIST/t10k-labels-idx1-ubyte')
    trainData = getImgs('MNIST/train-images-idx3-ubyte')
    testData = getImgs('MNIST/t10k-images-idx3-ubyte')
    trainLabel = np.array(trainLabels, dtype=np.int)
    testLabel = np.array(testLabels, dtype=np.int)
    trainData = np.array(trainData, dtype=np.float) / 255
    testData = np.array(testData, dtype=np.float) / 255
    np.savez('dataset',
            trainLabel = trainLabel,
            testLabel = testLabel,
            trainData = trainData,
            testData = testData)