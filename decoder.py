import argparse
parser = argparse.ArgumentParser()
import numpy as np


if __name__ == "__main__":
    parser.add_argument("--states",type=str)
    parser.add_argument("--value-policy",type=str)
    args = parser.parse_args()

    f_states = open(args.states)
    bbrr = [line.split()[0] for line in f_states]
    f_states.close()

    f_val_policy = open(args.value_policy)
    for state in bbrr:
        tmp = f_val_policy.readline().split()
        tmp[1] = [0,1,2,4,6][int(tmp[1])]
        print(state, tmp[1], float(tmp[0]))
    f_val_policy.close()
