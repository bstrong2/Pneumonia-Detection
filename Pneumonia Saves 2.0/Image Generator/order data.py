import os

fileLocation = os.getcwd() + '\\'

fileName = 'Confusion Matrix Info.txt'
fileOutput = 'Confusion Matrix Info_Final.txt'
fileOutputBinary = 'Confusion Matrix Info_Final_Binary.txt'
fileOutputFN = 'Confusion Matrix Info_Final_FN.txt'

fileNameBTS = 'Confusion Matrix Info Bigger Training Set.txt'
fileOutputBTS = 'Confusion Matrix Info_Final_BTS.txt'
fileOutputBinaryBTS = 'Confusion Matrix Info_Final_Binary_BTS.txt'
fileOutputFNBTS = 'Confusion Matrix Info_Final_FN_BTS.txt'


def read_data(file_name):

    model = []
    accuracy = []
    sensitivity = []
    specificity = []
    ppv = []
    npv = []
    binaryAcc = []
    precisionRate = []
    fn = []

    f = open(fileLocation+file_name, 'r')

    for i in f:

        l = i.split(' ')
        if 'Model:' in i:
            model.append(l[1].strip())

        elif 'Binary Accuracy:' in i:
            binaryAcc.append(l[2].strip())

        elif 'Accuracy:' in i:
            accuracy.append(l[1].strip())

        elif 'Sensitivity:' in i:
            sensitivity.append(l[1].strip())

        elif 'Specificity:' in i:
            specificity.append(l[1].strip())

        elif 'Positive Predictive Value:' in i:
            ppv.append(l[4].strip())

        elif 'Negative Predictive Value:' in i:
            npv.append(l[3].strip())

        elif 'Precision rate:' in i:
            precisionRate.append(l[2].strip())

        elif 'Num Of False Negatives:' in i:
            fn.append(l[4].strip())

    f.close()
    return model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn


model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn = read_data(fileName)


f2 = open(fileLocation+fileOutput, 'w')

for i in model:
    highest = 0.0
    index = 0
    counter = 0

    for i in accuracy:

        i = i.replace("%", "")
        if float(i) > highest:
            highest = float(i)
            index = counter

        counter += 1

    f2.write('Model: ' + str(model[index]) + '\n')
    f2.write('Accuracy: ' + str(accuracy[index]) + '\n')
    f2.write('Sensitivity: ' + str(sensitivity[index]) + '\n')
    f2.write('Specificity: ' + str(specificity[index]) + '\n')
    f2.write('Positive Predictive Value: ' + str(ppv[index]) + '\n')
    f2.write('Negative Predictive Value: ' + str(npv[index]) + '\n')
    f2.write('Binary Accuracy: ' + str(binaryAcc[index]) + '\n')
    f2.write('Precision rate: ' + str(precisionRate[index]) + '\n')
    f2.write('Num Of False Negatives: ' + str(fn[index]) + '\n\n')

    del model[index]
    del accuracy[index]
    del sensitivity[index]
    del specificity[index]
    del ppv[index]
    del npv[index]
    del binaryAcc[index]
    del precisionRate[index]
    del fn[index]

f2.close()


model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn = read_data(fileName)

f3 = open(fileLocation+fileOutputBinary, 'w')

for i in model:
    highest = 0.0
    index = 0
    counter = 0

    for i in binaryAcc:

        i = i.replace("%", "")
        if float(i) > highest:
            highest = float(i)
            index = counter

        counter += 1

    f3.write('Model: ' + str(model[index]) + '\n')
    f3.write('Accuracy: ' + str(accuracy[index]) + '\n')
    f3.write('Sensitivity: ' + str(sensitivity[index]) + '\n')
    f3.write('Specificity: ' + str(specificity[index]) + '\n')
    f3.write('Positive Predictive Value: ' + str(ppv[index]) + '\n')
    f3.write('Negative Predictive Value: ' + str(npv[index]) + '\n')
    f3.write('Binary Accuracy: ' + str(binaryAcc[index]) + '\n')
    f3.write('Precision rate: ' + str(precisionRate[index]) + '\n')
    f3.write('Num Of False Negatives: ' + str(fn[index]) + '\n\n')

    del model[index]
    del accuracy[index]
    del sensitivity[index]
    del specificity[index]
    del ppv[index]
    del npv[index]
    del binaryAcc[index]
    del precisionRate[index]
    del fn[index]

f3.close()

model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn = read_data(fileName)

f4 = open(fileLocation+fileOutputFN, 'w')

for i in fn:
    lowest = 9999999999999
    index = 0
    counter = 0

    for i in fn:

        if int(i) < lowest:
            lowest = int(i)
            index = counter

        counter += 1

    f4.write('Model: ' + str(model[index]) + '\n')
    f4.write('Accuracy: ' + str(accuracy[index]) + '\n')
    f4.write('Sensitivity: ' + str(sensitivity[index]) + '\n')
    f4.write('Specificity: ' + str(specificity[index]) + '\n')
    f4.write('Positive Predictive Value: ' + str(ppv[index]) + '\n')
    f4.write('Negative Predictive Value: ' + str(npv[index]) + '\n')
    f4.write('Binary Accuracy: ' + str(binaryAcc[index]) + '\n')
    f4.write('Precision rate: ' + str(precisionRate[index]) + '\n')
    f4.write('Num Of False Negatives: ' + str(fn[index]) + '\n\n')

    del model[index]
    del accuracy[index]
    del sensitivity[index]
    del specificity[index]
    del ppv[index]
    del npv[index]
    del binaryAcc[index]
    del precisionRate[index]
    del fn[index]

f4.close()


#
# Doing the same thing again, but with the bigger training set
#

model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn = read_data(fileNameBTS)


f2 = open(fileLocation+fileOutputBTS, 'w')

for i in model:
    highest = 0.0
    index = 0
    counter = 0

    for i in accuracy:

        i = i.replace("%", "")
        if float(i) > highest:
            highest = float(i)
            index = counter

        counter += 1

    f2.write('Model: ' + str(model[index]) + '\n')
    f2.write('Accuracy: ' + str(accuracy[index]) + '\n')
    f2.write('Sensitivity: ' + str(sensitivity[index]) + '\n')
    f2.write('Specificity: ' + str(specificity[index]) + '\n')
    f2.write('Positive Predictive Value: ' + str(ppv[index]) + '\n')
    f2.write('Negative Predictive Value: ' + str(npv[index]) + '\n')
    f2.write('Binary Accuracy: ' + str(binaryAcc[index]) + '\n')
    f2.write('Precision rate: ' + str(precisionRate[index]) + '\n')
    f2.write('Num Of False Negatives: ' + str(fn[index]) + '\n\n')

    del model[index]
    del accuracy[index]
    del sensitivity[index]
    del specificity[index]
    del ppv[index]
    del npv[index]
    del binaryAcc[index]
    del precisionRate[index]
    del fn[index]

f2.close()


model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn = read_data(fileNameBTS)

f3 = open(fileLocation+fileOutputBinaryBTS, 'w')

for i in model:
    highest = 0.0
    index = 0
    counter = 0

    for i in binaryAcc:

        i = i.replace("%", "")
        if float(i) > highest:
            highest = float(i)
            index = counter

        counter += 1

    f3.write('Model: ' + str(model[index]) + '\n')
    f3.write('Accuracy: ' + str(accuracy[index]) + '\n')
    f3.write('Sensitivity: ' + str(sensitivity[index]) + '\n')
    f3.write('Specificity: ' + str(specificity[index]) + '\n')
    f3.write('Positive Predictive Value: ' + str(ppv[index]) + '\n')
    f3.write('Negative Predictive Value: ' + str(npv[index]) + '\n')
    f3.write('Binary Accuracy: ' + str(binaryAcc[index]) + '\n')
    f3.write('Precision rate: ' + str(precisionRate[index]) + '\n')
    f3.write('Num Of False Negatives: ' + str(fn[index]) + '\n\n')

    del model[index]
    del accuracy[index]
    del sensitivity[index]
    del specificity[index]
    del ppv[index]
    del npv[index]
    del binaryAcc[index]
    del precisionRate[index]
    del fn[index]

f3.close()

model, accuracy, sensitivity, specificity, ppv, npv, binaryAcc, precisionRate, fn = read_data(fileNameBTS)

f4 = open(fileLocation+fileOutputFNBTS, 'w')

for i in fn:
    lowest = 9999999999999
    index = 0
    counter = 0

    for i in fn:

        if int(i) < lowest:
            lowest = int(i)
            index = counter

        counter += 1

    f4.write('Model: ' + str(model[index]) + '\n')
    f4.write('Accuracy: ' + str(accuracy[index]) + '\n')
    f4.write('Sensitivity: ' + str(sensitivity[index]) + '\n')
    f4.write('Specificity: ' + str(specificity[index]) + '\n')
    f4.write('Positive Predictive Value: ' + str(ppv[index]) + '\n')
    f4.write('Negative Predictive Value: ' + str(npv[index]) + '\n')
    f4.write('Binary Accuracy: ' + str(binaryAcc[index]) + '\n')
    f4.write('Precision rate: ' + str(precisionRate[index]) + '\n')
    f4.write('Num Of False Negatives: ' + str(fn[index]) + '\n\n')

    del model[index]
    del accuracy[index]
    del sensitivity[index]
    del specificity[index]
    del ppv[index]
    del npv[index]
    del binaryAcc[index]
    del precisionRate[index]
    del fn[index]

f4.close()
