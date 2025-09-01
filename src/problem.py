import numpy as np
import itertools
from typing import List, Sequence, Tuple


class PFSProblem:
    """
    Permutation Flow Shop (PFSP) problem utility.

    This class stores a processing-time matrix `ptime` of shape (nj, nm),
    where `nj` is the number of jobs and `nm` is the number of machines.
    """

    def __init__(self, ptime: np.ndarray) -> None:
        """
        Parameters
        ----------

        ptime : np.ndarray of shape (nj, nm), dtype integer
            Processing times: ptime[j, m] is the time for job j on machine m.
        """

        self.nj, self.nm = ptime.shape
        self.ptime = ptime

    @staticmethod
    def load(fname: str) -> "PFSProblem":
        """
        Load a PFS instance from a text file.

        Parameters
        ----------
        fname : str
            Path to the file.

        Returns
        -------
        PFSProblem
            The loaded problem instance.
        """
        with open(fname, "r") as f:
            f.readline()  # skip header/comment
            nj, nm = [int(x) for x in f.readline().strip().split()][:2]
            f.readline()  # skip header/comment
            ptime = np.zeros((nj, nm), dtype=np.int32)
            # The file lists times by machine (columns). We fill one column per line.
            for m in range(nm):
                ptime[:, m] = [int(x) for x in f.readline().strip().split()]
                
        return PFSProblem(ptime)

    @staticmethod
    def int2perm(x: int, n: int) -> List[int]:
        """
        Map an integer to a permutation of range(n) via mixed-radix (Lehmer-like) decoding.

        The algorithm repeatedly takes remainders with bases n, n-1, ..., 1:
            r_0 = x % n;             x //= n
            r_1 = x % (n-1);         x //= (n-1)
            ...
        Each remainder selects an element from the shrinking list of available items.

        Parameters
        ----------
        x : int
            Non-negative integer to decode.
        n : int
            Permutation size.

        Returns
        -------
        list[int]
            A permutation of 0..n-1.
        """
        l = list(range(n))  # available elements
        p: List[int] = []
        while n >= 1:
            r = x % n
            x //= n
            p.append(l[r])
            del l[r]
            n -= 1
        return p

    def evaluate(self, p: Sequence[int]) -> int:
        """
        Compute the makespan (C_max) of a given job permutation.

        Parameters
        ----------
        p : Sequence[int]
            A permutation of jobs (0..nj-1).

        Returns
        -------
        int
            Makespan (completion time of the last job on the last machine).
        """
        ctime = np.zeros((self.nj, self.nm), dtype=np.int32)

        # Initialize first job across machines.
        first_job = p[0]
        ctime[first_job, 0] = self.ptime[first_job, 0]
        for i in range(1, self.nm):
            ctime[first_job, i] = ctime[first_job, i - 1] + self.ptime[first_job, i]

        makespan = ctime[first_job, self.nm - 1]
        prev_job = first_job

        # Propagate through remaining jobs.
        for j in range(1, self.nj):
            job = p[j]

            # First machine: must wait for previous job on the same machine.
            ctime[job, 0] = ctime[prev_job, 0] + self.ptime[job, 0]

            # Remaining machines: classic flow-shop max-of-north-and-west recurrence.
            for i in range(1, self.nm):
                ctime[job, i] = max(ctime[job, i - 1], ctime[prev_job, i]) + self.ptime[job, i]

            makespan = max(makespan, ctime[job, self.nm - 1])
            prev_job = job

        return int(makespan)

    def evaluateb(self, x: str) -> int:
        """
        Evaluate the makespan of the permutation obtained from a binary string.

        Parameters
        ----------
        x : str
            Binary string, e.g., "101011".

        Returns
        -------
        int
            Makespan of the derived permutation.
        """
        k = PFSProblem.int2perm(int(x, 2), self.nj)
        return self.evaluate(k)

    def find_optima(self) -> Tuple[int, List[Tuple[int, ...]]]:
        """
        Brute-force the optimal makespan and collect all optimal permutations.

        Returns
        -------
        (fmin, opt) : (int, list[tuple[int, ...]])
            fmin : optimal makespan
            opt  : list of all optimal permutations achieving fmin (as tuples)
        """
        # Upper bound: sum of all processing times + 1 (strictly larger than any possible makespan).
        fmin = int(self.ptime.sum()) + 1
        opt: List[Tuple[int, ...]] = []

        # Enumerate all job permutations and keep the best.
        for c in itertools.permutations(range(self.nj)):
            f = self.evaluate(c)
            if f < fmin:
                fmin = f
                opt = [c]
            elif f == fmin:
                opt.append(c)

        return fmin, opt
