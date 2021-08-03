import sys


def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    hits = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            hits += 1
    return hits / len(y_true)


def f1_recall_precision(y_true, y_pred):
    labels = list(set(y_true))
    assert len(y_true) == len(y_pred)
    res = {}
    for label in labels:
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == label:
                    tp += 1
                else:
                    tn += 1
            else:
                if y_true[i] == label:
                    fn += 1
                else:
                    fp += 1

        recall = tp / (tp + fn + 1e-10)
        precison = tp / (tp + fp + 1e-10)
        f1 = 2 * precison * recall / (recall + precison + 1e-10)
        res[label] = {'f1': f1, 'recall': recall, 'precison': precison}
    return res


y_true = open(sys.argv[1], "r").readlines()
y_pre = open(sys.argv[2], "r").readlines()

y_true = list(map(int, y_true))
y_pre = list(map(int, y_pre))

assert len(y_true) == len(y_pre)

print(f1_recall_precision(y_true, y_pre))
print("accuracy:{}".format(accuracy(y_true, y_pre)))
