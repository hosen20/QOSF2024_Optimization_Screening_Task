from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import InequalityToEquality
from qiskit_optimization.converters import IntegerToBinary
from qiskit_optimization.converters import LinearEqualityToPenalty

from dimod import BQM


class Problem:
    """
    A class used to build problem equations.

    Attributes
    ----------
    num_trucks : int
        the number of trucks or bins in which the packets oor boxes are stored
    num_packets : int
        the number of packets or boxes to be stored in the bins or trucks
    weigths : int
        the weight of each packet
    c : int
        the maximum weight a bin or truck can hold

    Methods
    -------
    prepareProblemCplex()
        Builds the equations as BQM(binary quadratic model) to be used for VQE and QAOA

    prepareProblemDwave()
        Builds the equations as QUBO to be used for Dwave bruteforce solver and annealing solver

    toQuboCplex(qp)
        Transforms the BQM model from prepareProblemCplex() function into QUBO
    """
    def __init__(self, num_trucks, num_packets, weights, c):
        """
        Parameters
        ----------
        num_trucks : int
            the number of trucks or bins in which the packets oor boxes are stored
            
        num_packets : int
            the number of packets or boxes to be stored in the bins or trucks
            
        weigths : int
            the weight of each packet
            
        c : int
            the maximum weight a bin or truck can hold
        """
        
        self.num_trucks = num_trucks
        self.num_packets = num_packets
        self.weights = weights
        self.c = c

    
    def prepareProblemCplex(self):
        """ Builds the BQM model equations to be used in VQE and QAOA
        """
        
        qp = QuadraticProgram()

        ys = []
        xs = []

        for i in range(1, self.num_trucks+1):
            strng = "y"+str(i)
            ys.append([strng, qp.binary_var(strng)])

        for i in range(1, self.num_trucks+1):
            linear = {}
            for j in range(1, self.num_packets+1):
                strng = "x"+str(i)+str(j)
                xs.append([strng, qp.binary_var(strng)])
                linear[strng] = self.weights[j-1]
            linear[ys[i-1][0]] = -self.c
            qp.linear_constraint(linear=linear, sense="LE", rhs=0, name="xy"+str(i))

        for j in range(1, self.num_packets+1):
            linear = {}
            for x in xs:
                if x[0][2] == str(j):
                    linear[x[0]] = 1
            qp.linear_constraint(linear=linear, sense="E", rhs=1, name="x"+str(j))
    
        

        linear_to_minimize = {h[0]:1 for h in ys}
        qp.minimize(linear=linear_to_minimize)

        return qp

    
    def prepareProblemDwave(self):
        """ Builds the QUBO model equations to be used in D-wave bruteforce and annealing sampler
        """

        linear_obj = {}
        ys = []
        xs = []

        for i in range(1, self.num_trucks+1):
            strng = "y"+str(i)
            ys.append(strng)
            linear_obj[strng] = 1
        bqm = BQM(linear_obj, "BINARY")

        for i in range(1, self.num_trucks+1):
            c1 = []
            for j in range(1, self.num_packets+1):
                strng = "x"+str(i)+str(j)
                xs.append(strng)
                c1.append((strng, self.weights[j-1]))
            c1.append((ys[i-1], -self.c))
            bqm.add_linear_inequality_constraint(c1, label="c1_"+str(i), lb = -self.c, ub=0, lagrange_multiplier=6)

        for j in range(1, self.num_packets+1):
            c2 = []
            for x in xs:
                if x[2] == str(j):
                    c2.append((x, 1))
            bqm.add_linear_equality_constraint(c2, constant=-1, lagrange_multiplier=6)

        return bqm.to_qubo()


    def toQuboCplex(self, qp):
        """ Transforms the BQM model into QUBO model to be used in VQE and QAOA

            Parameters
            ----------
            qp : BQM
                The BQM model returned by prepareProblemCplex() function
        """
        
        ineq2eq = InequalityToEquality()
        qp_eq = ineq2eq.convert(qp)

        int2bin = IntegerToBinary()
        qp_eq_bin = int2bin.convert(qp_eq)

        lineq2penalty = LinearEqualityToPenalty()
        qubo = lineq2penalty.convert(qp_eq_bin)

        return qubo
    

        