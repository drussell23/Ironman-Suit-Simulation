"""
Neural Evolution for Iron Man Suit

This module provides evolutionary algorithms for neural network optimization:
- NeuroEvolution of Augmenting Topologies (NEAT)
- Genetic Algorithms for neural networks
- Evolutionary Strategies
- Coevolutionary algorithms
- Multi-objective evolution
- Adaptive mutation rates
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import random
import copy
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithms."""

    # General parameters
    population_size: int = 100
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 10

    # NEAT specific
    neat_compatibility_threshold: float = 3.0
    neat_weight_mutation_rate: float = 0.8
    neat_add_connection_rate: float = 0.05
    neat_add_node_rate: float = 0.03
    neat_connection_mutation_rate: float = 0.1

    # Genetic Algorithm specific
    ga_tournament_size: int = 3
    ga_uniform_crossover: bool = True

    # Evolutionary Strategies specific
    es_sigma: float = 0.1
    es_learning_rate: float = 0.01


class Gene:
    """Represents a gene in the genome."""

    def __init__(
        self,
        innovation_number: int,
        from_node: int,
        to_node: int,
        weight: float,
        enabled: bool = True,
    ):
        self.innovation_number = innovation_number
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled

    def copy(self) -> "Gene":
        """Create a copy of this gene."""
        return Gene(
            self.innovation_number,
            self.from_node,
            self.to_node,
            self.weight,
            self.enabled,
        )

    def mutate(self, mutation_rate: float, weight_mutation_rate: float) -> "Gene":
        """Mutate this gene."""
        gene = self.copy()

        # Weight mutation
        if random.random() < weight_mutation_rate:
            if random.random() < 0.1:  # 10% chance of completely new weight
                gene.weight = random.uniform(-1, 1)
            else:  # 90% chance of perturbing existing weight
                gene.weight += random.gauss(0, 0.1)
                gene.weight = np.clip(gene.weight, -1, 1)

        # Connection enable/disable
        if random.random() < mutation_rate:
            gene.enabled = not gene.enabled

        return gene


class Genome:
    """Represents a neural network genome in NEAT."""

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.genes: List[Gene] = []
        self.nodes: Dict[int, str] = {}  # node_id -> node_type
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

        # Initialize with input and output nodes
        for i in range(input_size):
            self.nodes[i] = "input"

        for i in range(output_size):
            self.nodes[input_size + i] = "output"

    def add_gene(self, gene: Gene):
        """Add a gene to the genome."""
        self.genes.append(gene)
        # Add nodes if they don't exist
        if gene.from_node not in self.nodes:
            self.nodes[gene.from_node] = "hidden"
        if gene.to_node not in self.nodes:
            self.nodes[gene.to_node] = "hidden"

    def copy(self) -> "Genome":
        """Create a copy of this genome."""
        new_genome = Genome(self.input_size, self.output_size)
        new_genome.genes = [gene.copy() for gene in self.genes]
        new_genome.nodes = self.nodes.copy()
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        return new_genome

    def mutate(
        self, config: EvolutionConfig, innovation_counter: int
    ) -> Tuple["Genome", int]:
        """Mutate this genome."""
        new_genome = self.copy()

        # Weight mutations
        for gene in new_genome.genes:
            if random.random() < config.neat_weight_mutation_rate:
                gene.mutate(config.mutation_rate, config.neat_weight_mutation_rate)

        # Add connection mutation
        if random.random() < config.neat_add_connection_rate:
            # Find two unconnected nodes
            available_nodes = list(new_genome.nodes.keys())
            for _ in range(20):  # Try 20 times to find valid connection
                from_node = random.choice(available_nodes)
                to_node = random.choice(available_nodes)

                # Check if connection already exists
                connection_exists = any(
                    g.from_node == from_node and g.to_node == to_node
                    for g in new_genome.genes
                )

                if not connection_exists and from_node != to_node:
                    # Check if this would create a cycle (simple check)
                    if not self._would_create_cycle(from_node, to_node):
                        new_gene = Gene(
                            innovation_counter,
                            from_node,
                            to_node,
                            random.uniform(-1, 1),
                        )
                        new_genome.add_gene(new_gene)
                        innovation_counter += 1
                        break

        # Add node mutation
        if random.random() < config.neat_add_node_rate:
            # Select a random enabled connection
            enabled_genes = [g for g in new_genome.genes if g.enabled]
            if enabled_genes:
                gene_to_split = random.choice(enabled_genes)
                gene_to_split.enabled = False

                # Create new node
                new_node_id = max(new_genome.nodes.keys()) + 1
                new_genome.nodes[new_node_id] = "hidden"

                # Create connections to and from new node
                new_gene1 = Gene(
                    innovation_counter, gene_to_split.from_node, new_node_id, 1.0
                )
                new_gene2 = Gene(
                    innovation_counter + 1,
                    new_node_id,
                    gene_to_split.to_node,
                    gene_to_split.weight,
                )

                new_genome.add_gene(new_gene1)
                new_genome.add_gene(new_gene2)
                innovation_counter += 2

        return new_genome, innovation_counter

    def _would_create_cycle(self, from_node: int, to_node: int) -> bool:
        """Check if adding a connection would create a cycle."""
        # Simple cycle detection using DFS
        visited = set()
        stack = [to_node]

        while stack:
            node = stack.pop()
            if node == from_node:
                return True

            if node not in visited:
                visited.add(node)
                for gene in self.genes:
                    if gene.enabled and gene.from_node == node:
                        stack.append(gene.to_node)

        return False

    def distance(self, other: "Genome") -> float:
        """Calculate distance between two genomes."""
        # Count disjoint and excess genes
        disjoint = 0
        excess = 0
        weight_diff = 0
        matching_genes = 0

        # Sort genes by innovation number
        self_genes = sorted(self.genes, key=lambda g: g.innovation_number)
        other_genes = sorted(other.genes, key=lambda g: g.innovation_number)

        i = j = 0
        while i < len(self_genes) and j < len(other_genes):
            if self_genes[i].innovation_number == other_genes[j].innovation_number:
                # Matching gene
                weight_diff += abs(self_genes[i].weight - other_genes[j].weight)
                matching_genes += 1
                i += 1
                j += 1
            elif self_genes[i].innovation_number < other_genes[j].innovation_number:
                # Disjoint gene in self
                disjoint += 1
                i += 1
            else:
                # Disjoint gene in other
                disjoint += 1
                j += 1

        # Count excess genes
        excess = len(self_genes) - i + len(other_genes) - j

        # Normalize by genome size
        max_size = max(len(self_genes), len(other_genes))
        if max_size < 20:
            max_size = 1

        distance = (disjoint + excess) / max_size
        if matching_genes > 0:
            distance += weight_diff / matching_genes

        return distance

    def to_neural_network(self) -> "NEATNetwork":
        """Convert genome to neural network."""
        return NEATNetwork(self)


class NEATNetwork(nn.Module):
    """Neural network implementation of a NEAT genome."""

    def __init__(self, genome: Genome):
        super().__init__()
        self.genome = genome
        self.node_order = self._topological_sort()

        # Create weight matrix
        max_node = max(genome.nodes.keys()) + 1
        self.weight_matrix = torch.zeros(max_node, max_node)

        # Set weights from genes
        for gene in genome.genes:
            if gene.enabled:
                self.weight_matrix[gene.from_node, gene.to_node] = gene.weight

    def _topological_sort(self) -> List[int]:
        """Perform topological sort of nodes."""
        # Find input nodes
        input_nodes = [
            node_id
            for node_id, node_type in self.genome.nodes.items()
            if node_type == "input"
        ]

        # Build adjacency list
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)

        for gene in self.genome.genes:
            if gene.enabled:
                adj_list[gene.from_node].append(gene.to_node)
                in_degree[gene.to_node] += 1

        # Kahn's algorithm
        queue = input_nodes.copy()
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        batch_size = x.size(0)
        max_node = self.weight_matrix.size(0)

        # Initialize node values
        node_values = torch.zeros(batch_size, max_node)

        # Set input values
        for i, node_id in enumerate(self.genome.nodes.keys()):
            if self.genome.nodes[node_id] == "input":
                node_values[:, node_id] = x[:, i]

        # Process nodes in topological order
        for node_id in self.node_order:
            if self.genome.nodes[node_id] == "input":
                continue

            # Compute weighted sum of inputs
            inputs = torch.matmul(node_values, self.weight_matrix[:, node_id])

            # Apply activation function
            if self.genome.nodes[node_id] == "output":
                node_values[:, node_id] = torch.tanh(inputs)
            else:
                node_values[:, node_id] = torch.relu(inputs)

        # Extract output values
        output_nodes = [
            node_id
            for node_id, node_type in self.genome.nodes.items()
            if node_type == "output"
        ]
        outputs = node_values[:, output_nodes]

        return outputs


class NEAT:
    """NeuroEvolution of Augmenting Topologies (NEAT) implementation."""

    def __init__(self, input_size: int, output_size: int, config: EvolutionConfig):
        self.config = config
        self.input_size = input_size
        self.output_size = output_size

        # Population
        self.population: List[Genome] = []
        self.species: List[List[Genome]] = []

        # Innovation tracking
        self.innovation_counter = 0
        self.innovation_history: Dict[Tuple[int, int], int] = {}

        # Statistics
        self.generation = 0
        self.best_fitness = float("-inf")
        self.best_genome = None

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Initialize the population with minimal genomes."""
        for _ in range(self.config.population_size):
            genome = Genome(self.input_size, self.output_size)

            # Add connections from each input to each output
            for input_node in range(self.input_size):
                for output_node in range(self.output_size):
                    output_node_id = self.input_size + output_node
                    gene = Gene(
                        self.innovation_counter,
                        input_node,
                        output_node_id,
                        random.uniform(-1, 1),
                    )
                    genome.add_gene(gene)
                    self.innovation_counter += 1

            self.population.append(genome)

    def evaluate_fitness(self, fitness_function: Callable[[Genome], float]):
        """Evaluate fitness of all genomes in the population."""
        for genome in self.population:
            genome.fitness = fitness_function(genome)

            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()

    def speciate(self):
        """Group genomes into species based on compatibility."""
        self.species = []

        for genome in self.population:
            # Find compatible species
            placed = False
            for species in self.species:
                if species:  # Check if species is not empty
                    representative = species[0]
                    if (
                        genome.distance(representative)
                        < self.config.neat_compatibility_threshold
                    ):
                        species.append(genome)
                        placed = True
                        break

            # Create new species if no compatible species found
            if not placed:
                self.species.append([genome])

    def adjust_fitness(self):
        """Adjust fitness based on species size."""
        for species in self.species:
            if species:
                species_size = len(species)
                for genome in species:
                    genome.adjusted_fitness = genome.fitness / species_size

    def select_parents(self) -> Tuple[Genome, Genome]:
        """Select parents for crossover using tournament selection."""
        # Select from random species
        non_empty_species = [s for s in self.species if s]
        if not non_empty_species:
            return random.choice(self.population), random.choice(self.population)

        species1 = random.choice(non_empty_species)
        species2 = random.choice(non_empty_species)

        # Tournament selection within species
        parent1 = self._tournament_select(species1)
        parent2 = self._tournament_select(species2)

        return parent1, parent2

    def _tournament_select(self, species: List[Genome]) -> Genome:
        """Tournament selection within a species."""
        tournament = random.sample(species, min(3, len(species)))
        return max(tournament, key=lambda g: g.adjusted_fitness)

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Perform crossover between two parents."""
        # Ensure parent1 is more fit
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        child = Genome(self.input_size, self.output_size)

        # Inherit genes from parents
        for gene1 in parent1.genes:
            # Find matching gene in parent2
            matching_gene = None
            for gene2 in parent2.genes:
                if gene2.innovation_number == gene1.innovation_number:
                    matching_gene = gene2
                    break

            if matching_gene:
                # Inherit from either parent
                if random.random() < 0.5:
                    child.add_gene(gene1.copy())
                else:
                    child.add_gene(matching_gene.copy())
            else:
                # Disjoint gene - inherit from more fit parent
                child.add_gene(gene1.copy())

        return child

    def evolve(self, fitness_function: Callable[[Genome], float]) -> Dict[str, Any]:
        """Perform one generation of evolution."""
        # Evaluate fitness
        self.evaluate_fitness(fitness_function)

        # Speciate
        self.speciate()

        # Adjust fitness
        self.adjust_fitness()

        # Create new population
        new_population = []

        # Elitism - keep best genome from each species
        for species in self.species:
            if species:
                best_genome = max(species, key=lambda g: g.fitness)
                new_population.append(best_genome.copy())

        # Fill rest of population
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
            else:
                # Clone
                parent = self._tournament_select(
                    random.choice([s for s in self.species if s])
                )
                child = parent.copy()

            # Mutate
            child, self.innovation_counter = child.mutate(
                self.config, self.innovation_counter
            )
            new_population.append(child)

        # Update population
        self.population = new_population[: self.config.population_size]
        self.generation += 1

        # Calculate statistics
        avg_fitness = np.mean([g.fitness for g in self.population])
        max_fitness = max(g.fitness for g in self.population)
        num_species = len([s for s in self.species if s])

        return {
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "best_fitness": self.best_fitness,
            "num_species": num_species,
            "population_size": len(self.population),
        }


class GeneticAlgorithm:
    """Genetic Algorithm for neural network optimization."""

    def __init__(self, model_class: type, model_params: Dict, config: EvolutionConfig):
        self.config = config
        self.model_class = model_class
        self.model_params = model_params

        # Population
        self.population: List[nn.Module] = []
        self.fitness_scores: List[float] = []

        # Statistics
        self.generation = 0
        self.best_fitness = float("-inf")
        self.best_individual = None

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Initialize the population with random individuals."""
        for _ in range(self.config.population_size):
            individual = self.model_class(**self.model_params)
            self.population.append(individual)
            self.fitness_scores.append(0.0)

    def evaluate_fitness(self, fitness_function: Callable[[nn.Module], float]):
        """Evaluate fitness of all individuals in the population."""
        for i, individual in enumerate(self.population):
            self.fitness_scores[i] = fitness_function(individual)

            if self.fitness_scores[i] > self.best_fitness:
                self.best_fitness = self.fitness_scores[i]
                self.best_individual = copy.deepcopy(individual)

    def select_parents(self) -> Tuple[nn.Module, nn.Module]:
        """Select parents using tournament selection."""
        parent1 = self._tournament_select()
        parent2 = self._tournament_select()
        return parent1, parent2

    def _tournament_select(self) -> nn.Module:
        """Tournament selection."""
        tournament_indices = random.sample(
            range(len(self.population)), self.config.ga_tournament_size
        )
        best_index = max(tournament_indices, key=lambda i: self.fitness_scores[i])
        return copy.deepcopy(self.population[best_index])

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        """Perform crossover between two parents."""
        child = copy.deepcopy(parent1)

        # Uniform crossover of parameters
        for (name1, param1), (name2, param2) in zip(
            parent1.named_parameters(), parent2.named_parameters()
        ):
            if random.random() < 0.5:
                child_param = child.get_parameter(name1)
                child_param.data = param2.data.clone()

        return child

    def mutate(self, individual: nn.Module):
        """Mutate an individual."""
        for param in individual.parameters():
            if random.random() < self.config.mutation_rate:
                # Add Gaussian noise
                noise = torch.randn_like(param.data) * 0.1
                param.data += noise

    def evolve(self, fitness_function: Callable[[nn.Module], float]) -> Dict[str, Any]:
        """Perform one generation of evolution."""
        # Evaluate fitness
        self.evaluate_fitness(fitness_function)

        # Create new population
        new_population = []

        # Elitism - keep best individuals
        elite_indices = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )[: self.config.elite_size]

        for idx in elite_indices:
            new_population.append(copy.deepcopy(self.population[idx]))

        # Fill rest of population
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
            else:
                # Clone
                parent = self._tournament_select()
                child = copy.deepcopy(parent)

            # Mutate
            self.mutate(child)
            new_population.append(child)

        # Update population
        self.population = new_population[: self.config.population_size]
        self.fitness_scores = [0.0] * len(self.population)
        self.generation += 1

        # Calculate statistics
        avg_fitness = np.mean(self.fitness_scores) if self.fitness_scores else 0.0
        max_fitness = max(self.fitness_scores) if self.fitness_scores else 0.0

        return {
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "best_fitness": self.best_fitness,
            "population_size": len(self.population),
        }


class EvolutionaryStrategies:
    """Evolutionary Strategies for neural network optimization."""

    def __init__(self, model_class: type, model_params: Dict, config: EvolutionConfig):
        self.config = config
        self.model_class = model_class
        self.model_params = model_params

        # Population
        self.population: List[nn.Module] = []
        self.fitness_scores: List[float] = []

        # Strategy parameters
        self.sigma = config.es_sigma
        self.learning_rate = config.es_learning_rate

        # Statistics
        self.generation = 0
        self.best_fitness = float("-inf")
        self.best_individual = None

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Initialize the population."""
        for _ in range(self.config.population_size):
            individual = self.model_class(**self.model_params)
            self.population.append(individual)
            self.fitness_scores.append(0.0)

    def evaluate_fitness(self, fitness_function: Callable[[nn.Module], float]):
        """Evaluate fitness of all individuals in the population."""
        for i, individual in enumerate(self.population):
            self.fitness_scores[i] = fitness_function(individual)

            if self.fitness_scores[i] > self.best_fitness:
                self.best_fitness = self.fitness_scores[i]
                self.best_individual = copy.deepcopy(individual)

    def evolve(self, fitness_function: Callable[[nn.Module], float]) -> Dict[str, Any]:
        """Perform one generation of evolution."""
        # Evaluate fitness
        self.evaluate_fitness(fitness_function)

        # Sort population by fitness
        sorted_indices = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )

        # Calculate weights for weighted recombination
        weights = np.array(
            [
                max(0, np.log(self.config.population_size / 2 + 1) - np.log(i + 1))
                for i in range(self.config.population_size)
            ]
        )
        weights = weights / np.sum(weights)

        # Update best individual
        best_individual = copy.deepcopy(self.population[sorted_indices[0]])

        # Weighted recombination of parameters
        for param_name, param in best_individual.named_parameters():
            weighted_sum = torch.zeros_like(param.data)

            for i, idx in enumerate(sorted_indices):
                individual = self.population[idx]
                individual_param = individual.get_parameter(param_name)
                weighted_sum += weights[i] * individual_param.data

            param.data = weighted_sum

        # Update population with perturbed versions of best individual
        new_population = [best_individual]

        for _ in range(self.config.population_size - 1):
            individual = copy.deepcopy(best_individual)

            # Add noise to parameters
            for param in individual.parameters():
                noise = torch.randn_like(param.data) * self.sigma
                param.data += noise

            new_population.append(individual)

        # Update population
        self.population = new_population
        self.fitness_scores = [0.0] * len(self.population)
        self.generation += 1

        # Update strategy parameters
        self.sigma *= 0.99  # Decay sigma

        # Calculate statistics
        avg_fitness = np.mean(self.fitness_scores) if self.fitness_scores else 0.0
        max_fitness = max(self.fitness_scores) if self.fitness_scores else 0.0

        return {
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "best_fitness": self.best_fitness,
            "sigma": self.sigma,
            "population_size": len(self.population),
        }
