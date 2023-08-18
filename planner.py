import argparse
parser = argparse.ArgumentParser()
import numpy as np
import pulp

class my_MDP:
    def __init__(self, path):
        
        self.S = 0
        self.A = 0
        self.T = None
        self.R = None
        self.gamma = 0
        self.end_states = []
        self.mdtype = ""

        file = open(path,"r")
        for line in file:
            tokens = line.split()
            if tokens[0] == "numStates":
                self.S = int(tokens[1])
                continue
            if tokens[0] == "numActions":
                self.A = int(tokens[1])
                self.T = np.zeros((self.S, self.A, self.S))
                self.R = np.zeros((self.S, self.A, self.S))
                continue
            if tokens[0] == "end":
                if tokens[1] != "-1":
                    self.end_states = np.array(tokens[1:], dtype=np.uint32)
                continue
            if tokens[0] == "transition":
                s = int(tokens[1])
                a = int(tokens[2])
                s_ = int(tokens[3])
                r = float(tokens[4])
                t = float(tokens[5])
                if t == 0:
                    continue
                """
                    modified to handle situation where s,a,s_ is present more than once in the mdp file
                    enables us to encode conveniently, if s,a->s_ can happen in multiple ways (each with a r,t)
                    T keeps adding up
                    R needs to be calculated according to bayes
                    For our encoding, R only depends on s_ so it's same in all those ways and this calculation is not necessary
                """
                # self.R[s,a,s_] = r
                self.R[s,a,s_] *= self.T[s,a,s_]    # mod
                self.R[s,a,s_] += r*t               # mod
                self.T[s,a,s_] += t
                self.R[s,a,s_] /= self.T[s,a,s_]    # mod
                continue
            if tokens[0] == "mdtype":
                self.mdtype = tokens[1]
                continue
            if tokens[0] == "discount":
                self.gamma = float(tokens[1])
                continue
        file.close()

        if self.mdtype == "episodic":
            for state in self.end_states:
                self.T[state,:,state] = 1
                self.R[state,:,state] = 0

        self.TR = (self.T*self.R).sum(axis = 2) # sum_{s'}  T.R (s,a,s')

    def V_pi(self,pi):
        T_ = self.T[np.arange(self.S),pi,:] # T(s,pi(s),s')
        er0 = self.TR[np.arange(self.S),pi] # E[r^0]
        # we have (I - gamma T_) @ V_pi = er0
        return np.linalg.solve( np.eye(self.S) - self.gamma*T_ , er0)

    def pi_star_from_v_star(self, V_star):
        Q = self.TR + self.gamma*(self.T@V_star)    # 3D @ 1D matmul operation happens to be correct
        return np.argmax(Q, axis = 1)

    def B_star(self,V):
        Q = self.TR + self.gamma*(self.T@V)         # 3D @ 1D matmul operation happens to be correct
        return np.max(Q, axis = 1)

    def vi(self):
        V_star = np.zeros(self.S)
        while True:
            new = self.B_star(V_star)
            if np.linalg.norm(new-V_star) < 1e-8:
                break
            else:
                V_star = new
        pi_star = self.pi_star_from_v_star(V_star)
        return V_star,pi_star

    def lp(self):
        model = pulp.LpProblem("mdp-solver", pulp.LpMinimize)
        
        V = [pulp.LpVariable(f"v{i}") for i in range(self.S)]

        model += pulp.lpSum(V), "minimize element sum of V"
        for s in range(self.S):
            for a in range(self.A):
                model += self.TR[s,a] + pulp.lpDot(self.gamma*self.T[s,a,:], V) <= V[s], f"s,a = {s},{a}"
        
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        V_star = np.array([V[i].varValue for i in range(self.S)])
        pi_star = self.pi_star_from_v_star(V_star)
        
        return V_star,pi_star

    def dual_lp(self):      # Littman et al. (1995)
        model = pulp.LpProblem("mdp-solver", pulp.LpMaximize)
        
        X = [[pulp.LpVariable(f"x{i},{j}", 0) for j in range(self.A)] for i in range(self.S)]
        X = np.array(X, dtype=object)
        
        model += pulp.lpDot(self.TR.reshape(-1), X.reshape(-1)), "maximize"
        for s in range(self.S):
            model += pulp.lpSum(X[s,:]) - 1 == self.gamma*pulp.lpDot(self.T[:,:,s].reshape(-1), X.reshape(-1)), f"s = {s}"
        
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        X_values = np.array([[X[i,j].varValue for j in range(self.A)] for i in range(self.S)])
        pi_star = np.argmax(X_values, axis = 1)
        V_star = self.V_pi(pi_star)
        
        return V_star,pi_star

    def hpi(self):
        pi = np.zeros(self.S, dtype=np.uint32)
        while True:
            vpi = self.V_pi(pi)
            Qpi = self.TR + self.gamma*(self.T@vpi)
            new = np.argmax(Qpi, axis = 1)  # improving action with max Q is chosen, all improvable states are improved
            if (pi == new).all():   # no states were improvable
                break
            else:
                pi = new
        return vpi,pi


if __name__ == "__main__":
    parser.add_argument("--mdp",type=str)
    parser.add_argument("--algorithm",type=str,default="hpi")
    parser.add_argument("--policy",type=str,default="NA")
    args = parser.parse_args()

    np.set_printoptions(linewidth=200)

    mdp = my_MDP(args.mdp)

    if args.policy != "NA":
        file = open(args.policy,"r")
        Pi = [int(line) for line in file]
        file.close()
        V = mdp.V_pi(Pi)
    else:
        if args.algorithm == "vi":
            V,Pi = mdp.vi()
        if args.algorithm == "lp":
            V,Pi = mdp.lp()
        if args.algorithm == "dual_lp":
            V,Pi = mdp.dual_lp()
        if args.algorithm == "hpi":
            V,Pi = mdp.hpi()

    for i in range(mdp.S):
        print(V[i], Pi[i])




