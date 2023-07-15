from __future__ import annotations

from copy import deepcopy
from pickle import dumps
from random import shuffle, sample
from sys import getsizeof
from typing import List, Any, Tuple
from zlib import compress

from chapter05.chromosome import Chromosome
from chapter05.generic_algorithm import GeneticAlgorithm

PEOPLE: List[str] = ["Michael", "Sarah", "Joshua", "Narine", "David", "Sajid",
                     "Melanie", "Daniel", "Wei", "Dean", "Brian", "Murant", "Lisa"]


class ListCompossion(Chromosome):
    def __init__(self, lst: List[Any]) -> None:
        self.lst: List[Any] = lst

    @property
    def byte_compressed(self) -> int:
        return getsizeof(compress(dumps(self.lst)))

    def fitness(self) -> float:
        return 1 / self.byte_compressed

    @classmethod
    def random_instance(cls) -> ListCompossion:
        mylst: List[str] = deepcopy(PEOPLE)
        shuffle(mylst)
        return ListCompossion(mylst)

    def crossover(self, other: ListCompossion) -> Tuple[ListCompossion, ListCompossion]:
        child1: ListCompossion = deepcopy(self)
        child2: ListCompossion = deepcopy(other)
        idx1, idx2 = sample(range(len(self.lst)), k=2)
        l1, l2 = child1.lst[idx1], child2.lst[idx2]
        child1.lst[child1.lst.index(l2)], child1.lst[idx2] = child1.lst[idx2], l2
        child2.lst[child2.lst.index(l1)], child2.lst[idx1] = child2.lst[idx1], l1
        return child1, child2

    def mutate(self) -> None:
        idx1, idx2 = sample(range(len(self.lst)), k=2)
        self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]

    def __str__(self) -> str:
        return f"Order: {self.lst} Bytes: {self.byte_compressed}"


if __name__ == '__main__':
    initial_population: List[ListCompossion] = [ListCompossion.random_instance() for _ in range(1000)]
    ga: GeneticAlgorithm[ListCompossion] = GeneticAlgorithm(initial_population=initial_population, threshold=1.0,
                                                            max_generations=1000, mutation_chance=0.2,
                                                            crossover_chance=0.7,
                                                            selection_type=GeneticAlgorithm.SelectionType.TOURNAMENT)
    result: ListCompossion = ga.run()
    print(result)
