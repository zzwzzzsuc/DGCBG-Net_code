import numpy as np

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def k_fold_data(k, i, X, y, pet, bou):
    assert k > 1

    Xy = list(zip(X, y, pet, bou))
    np.random.shuffle(Xy)  # shuffle
    X[:], y[:], pet[:], bou[:] = zip(*Xy)

    fold_size = len(X) // k

    X_train, y_train, pet_train, bou_train = None, None, None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part, pet_part, bou_part = X[idx], y[idx], pet[idx], bou[idx]

        if j == i:
            X_valid, y_valid, pet_valid, bou_valid = X_part, y_part, pet_part, bou_part
        elif X_train is None:
            X_train, y_train, pet_train, bou_train = X_part, y_part, pet_part, bou_part
        else:
            X_train = X_train + X_part
            y_train = y_train + y_part
            pet_train = pet_train + pet_part
            bou_train = bou_train + bou_part

    return X_train, y_train, X_valid, y_valid, pet_valid, pet_train, bou_train, bou_valid