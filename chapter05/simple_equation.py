from __future__ import annotations

from copy import deepcopy
from random import randrange, random
from typing import Tuple, List

from chapter05.chromosome import Chromosome
from chapter05.generic_algorithm import GeneticAlgorithm


class SimpleEquation(Chromosome):
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def fitness(self) -> float:  # 6x - x^2 + 4y - y^2
        return 6 * self.x - self.x ** 2 + 4 * self.y - self.y ** 2

    @classmethod
    def random_instance(cls) -> SimpleEquation:
        return SimpleEquation(randrange(100), randrange(100))

    def crossover(self, other: SimpleEquation) -> Tuple[SimpleEquation, SimpleEquation]:
        child1: SimpleEquation = deepcopy(self)
        child2: SimpleEquation = deepcopy(other)
        child1.y = other.y
        child2.y = self.y
        return child1, child2

    def mutate(self) -> None:
        if random() > 0.5:
            if random() > 0.5:
                self.x += 1
            else:
                self.x -= 1
        else:
            if random() > 0.5:
                self.y += 1
            else:
                self.y -= 1

    def __str__(self) -> str:
        return f"X: {self.x} Y: {self.y} Fitness: {self.fitness()}"


if __name__ == '__main__':
    initial_population: List[SimpleEquation] = [SimpleEquation.random_instance() for _ in range(20)]
    ga: GeneticAlgorithm[SimpleEquation] = GeneticAlgorithm(initial_population=initial_population, threshold=13.0,
                                                            max_generations=100, mutation_chance=0.1,
                                                            crossover_chance=0.7)
    result: SimpleEquation = ga.run()
    print(result)