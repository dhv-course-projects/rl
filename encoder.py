import argparse
parser = argparse.ArgumentParser()
import numpy as np


if __name__ == "__main__":
    parser.add_argument("--states",type=str)
    parser.add_argument("--parameters",type=str)
    parser.add_argument("--q",type=str,help="Weakness of player B", default="0.25")
    args = parser.parse_args()

    q = float(args.q)

    f_states = open(args.states)
    bbrr = [(int(line[0:2]), int(line[2:4])) for line in f_states]
    f_states.close()
    
    f_parameters = open(args.parameters)
    f_parameters.readline()
    param = np.array([f_parameters.readline().split()[1:] for i in range(5)], dtype=np.float32)
    f_parameters.close()

    B = bbrr[0][0]
    R = bbrr[0][1]
    win_state = B*R*2
    loss_state = B*R*2 + 1

    print("numStates", B*R*2 + 2)
    print("numActions", 5)
    print("end", win_state, loss_state)
    
    for state in range(B*R):        # states with A facing the ball
        for action in range(5):
            for result in range(7):
                prob = param[action][result]
                result = [-1,0,1,2,3,4,6][result]   # -1,0,1,2,3,4,6
                if prob > 0:
                    if(result == -1):
                        print("transition", state, action, loss_state, 0, prob)
                        continue
                    if(result >= bbrr[state][1]):
                        print("transition", state, action, win_state, 1, prob)
                        continue
                    if(bbrr[state][0] == 1):
                        print("transition", state, action, loss_state, 0, prob)
                        continue
                    new_state = state + R + result + B*R*((result == 1 or result == 3) ^ (bbrr[state][0]%6 == 1))
                    print("transition", state, action, new_state, 0, prob)

    for state in range(B*R, B*R*2): # states with B facing the ball
        for result in range(3):
            prob = [q, (1-q)/2, (1-q)/2 ][result]
            result = [-1,0,1][result]           # -1,0,1
            if(result == -1):
                for action in range(5):
                    print("transition", state, action, loss_state, 0, prob)
                continue
            if(result >= bbrr[state-B*R][1]):
                for action in range(5):
                    print("transition", state, action, win_state, 1, prob)
                continue
            if(bbrr[state-B*R][0] == 1):
                for action in range(5):
                    print("transition", state, action, loss_state, 0, prob)
                continue
            new_state = state + R + result - B*R*((result == 1) ^ (bbrr[state-B*R][0]%6 == 1))
            for action in range(5):
                print("transition", state, action, new_state, 0, prob)

    # for state in range(B*R):        # states with A facing the ball
    #     for action in range(5):
    #         def transition(s, p):
    #             print("transition", state, action, s, int(s == win_state), p)
    #             return
    #         for result in range(7):
    #             prob = param[action][result]
    #             result = [-1,0,1,2,3,4,6][result]   # -1,0,1,2,3,4,6
    #             if prob > 0:
    #                 if(result == -1):
    #                     transition(loss_state, prob)
    #                     continue
    #                 if(result >= bbrr[state][1]):
    #                     transition(win_state, prob)
    #                     continue
    #                 if(bbrr[state][0] == 1):
    #                     transition(loss_state, prob)
    #                     continue
                    
    #                 new_state = state + R + result
    #                 strike_change = ((result == 1 or result == 3) ^ (bbrr[state][0]%6 == 1))
                    
    #                 if not strike_change:
    #                     transition(new_state, prob)
    #                 else:
    #                     b = bbrr[new_state][0]
    #                     r = bbrr[new_state][1]
    #                     while b >= 1 :
    #                         transition(loss_state, prob*q)                          # -1
    #                         if b == 1:
    #                             if r > 1:
    #                                 transition(loss_state, prob*(1-q))              # 0,1
    #                             else:
    #                                 transition(loss_state, prob*((1-q)/2))          # 0
    #                                 transition(win_state, prob*((1-q)/2))           # 1
    #                             break
    #                         else:
    #                             over_change = (b%6 == 1)
    #                             if r == 1:
    #                                 transition(win_state, prob*((1-q)/2))           # 1
    #                                 if over_change:
    #                                     transition(new_state + R , prob*((1-q)/2))  # 0
    #                                     break
    #                                 else:
    #                                     prob *= ((1-q)/2)
    #                                     b -= 1
    #                                     new_state += R
    #                                     continue                                    # 0
    #                             else :
    #                                 if over_change:
    #                                     transition(new_state+R, prob*((1-q)/2))     # 0
    #                                     prob *= ((1-q)/2)
    #                                     b -= 1
    #                                     r -= 1
    #                                     new_state += R+1
    #                                     continue                                    # 1
    #                                 else:
    #                                     transition(new_state+R+1, prob*((1-q)/2))   # 1
    #                                     prob *= ((1-q)/2)
    #                                     b -= 1
    #                                     new_state += R
    #                                     continue                                    # 0

    print("mdptype", "episodic")
    print("discount", 1)

