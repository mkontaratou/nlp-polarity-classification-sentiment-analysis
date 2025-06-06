import time, sys
import numpy as np
import argparse

import torch

from classifier import Classifier


def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)



def load_label_output(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        return [line.strip().split("\t")[0] for line in f if line.strip()]



def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    return acc*100



def train_and_eval(classifier: Classifier, trainfile: str, devfile: str, testfile: str, run_id: int, device):
    print(f"\nRUN: {run_id}")
    print("  %s.1. Training the classifier..." % str(run_id))
    classifier.train(trainfile, devfile, device)
    print()
    print("  %s.2. Eval on the dev set..." % str(run_id), end="")
    slabels = classifier.predict(devfile, device)
    glabels = load_label_output(devfile)
    devacc = eval_list(glabels, slabels)
    print(" Acc.: %.2f" % devacc)
    testacc = -1
    if testfile is not None:
        # Evaluation on the test data
        print("  %s.3. Eval on the test set..." % str(run_id), end="")
        slabels = classifier.predict(testfile, device)
        glabels = load_label_output(testfile)
        testacc = eval_list(glabels, slabels)
        print(" Acc.: %.2f" % testacc)
    print()
    return (devacc, testacc)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--n_runs', help='Number of runs.', type=int, default=5)
    argparser.add_argument('-g', '--gpu', help='GPU device id on which to run the model', type=int)
    argparser.add_argument('-o', '--ollama_url', help='Full URL of the ollama server (including port number), if any', type=str)
    args = argparser.parse_args()
    device_name = "cpu" if args.gpu is None else f"cuda:{args.gpu}"
    device = torch.device(device_name)
    n_runs = args.n_runs
    set_reproducible()
    datadir = "../data/"
    trainfile =  datadir + "traindata.csv"
    devfile =  datadir + "devdata.csv"
    testfile = None
    # testfile = datadir + "testdata.csv"

    # Runs
    start_time = time.perf_counter()
    devaccs = []
    testaccs = []
    for i in range(1, n_runs+1):
        classifier =  Classifier(ollama_url=args.ollama_url)
        devacc, testacc = train_and_eval(classifier, trainfile, devfile, testfile, i, device)
        devaccs.append(np.round(devacc,2))
        testaccs.append(np.round(testacc,2))
    print('\nCompleted %d runs.' % n_runs)
    total_exec_time = (time.perf_counter() - start_time)
    print("Dev accs:", devaccs)
    print("Test accs:", testaccs)
    print()
    print("Mean Dev Acc.: %.2f (%.2f)" % (np.mean(devaccs), np.std(devaccs)))
    print("Mean Test Acc.: %.2f (%.2f)" % (np.mean(testaccs), np.std(testaccs)))
    print("\nExec time: %.2f s. ( %d per run )" % (total_exec_time, total_exec_time / n_runs))






