from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from dwave_qbsolv import QBSolv
import dimod

class Experiment:
    """
    A class used to build and run experiments.

    Attributes
    ----------
    qubo_vqe_qaoa : QUBO
        the QUBO model to be used in running VQE and QAOA
        
    qubo_dwave : QUBO
        the QUBO model to be used in D-wave bruteforce and annealing sampler

    Methods
    -------
    runVqe(max_iter, ansatz)
        Solves qubo_vqe_qaoa model using VQE

    runQAOA(max_iter, initial_point)
        Solves qubo_vqe_qaoa model using QAOA

    bruteForce()
        Solves qubo_dwave model using bruteforce approach

    annealer()
        Solves qubo_dwave model using annealing approach
    """
    
    def __init__(self, qubo_vqe_qaoa, qubo_dwave):
        """
        Parameters
        ----------
        qubo_vqe_qaoa : QUBO
            the QUBO model to be used in running VQE and QAOA
        
        qubo_dwave : QUBO
            the QUBO model to be used in D-wave bruteforce and annealing sampler
        """
        
        self.qubo_vqe_qaoa = qubo_vqe_qaoa
        self.qubo_dwave = qubo_dwave

    def runVqe(self, max_iter, ansatz):
        """ Solves qubo_vqe_qaoa model using VQE

            Parameters
            ----------
            max_iter : int
                The number of iterations performed by COBYLA

            ansatz : QuantumCircuit
                The trainable circuit by VQE
        """
        
        vqe = SamplingVQE(sampler=Sampler(), optimizer=COBYLA(maxiter=max_iter), ansatz = ansatz)
        vqe_optimizer = MinimumEigenOptimizer(vqe)
        vqe_result = vqe_optimizer.solve(self.qubo_vqe_qaoa)
        return vqe_result

    def runQAOA(self, max_iter, initial_point):
        """ Solves qubo_vqe_qaoa model using QAOA

            Parameters
            ----------
            max_iter : int
                The number of iterations performed by COBYLA

            initial_point : list
                The starting state of QAOA
        """
        
        qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=max_iter), initial_point=initial_point)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        qaoa_result = qaoa_optimizer.solve(self.qubo_vqe_qaoa)
        return qaoa_result

    def bruteForce(self):
        """ Solves qubo_dwave using a bruteforce approach
        """
        
        result = dimod.reference.samplers.ExactSolver().sample_qubo(self.qubo_dwave[0])
        return  result


    def annealer(self):
        """ Solves qubo_dwave using an annealing approach
        """
        
        result = QBSolv().sample_qubo(self.qubo_dwave[0])
        return result

        
        
        