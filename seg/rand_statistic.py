def rand_statistic(xGroundTruth, xPredicted):
    tDict = {'TP':0,'FP':0, 'FN':0, 'TN':0}
    for i in range(len(xPredicted)):
        for j in range(i + 1, len(xPredicted)):
            if xGroundTruth[i] == xGroundTruth[j] and xPredicted[i] == xPredicted[j]:
                tDict['TP'] += 1
            elif xGroundTruth[i] == xGroundTruth[j] and xPredicted[i] != xPredicted[j]:
                tDict['FN'] += 1
            elif xGroundTruth[i] != xGroundTruth[j] and xPredicted[i] == xPredicted[j]:
                tDict['FP'] += 1
            elif xGroundTruth[i] != xGroundTruth[j] and xPredicted[i] != xPredicted[j]:
                tDict['TN'] += 1
    rand = float(tDict['TP'] + tDict['TN']) / float(tDict['TP'] + tDict['TN'] + tDict['FN'] + tDict['FP'])
    return rand