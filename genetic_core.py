import random
import math
import mpmath
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import Config
import sympy as sp
import re

@dataclass
class MathExpression:
    """Represents a mathematical expression candidate"""
    expression: str
    value: Optional[float] = None
    error: Optional[float] = None
    complexity: int = 0
    elegance_score: Optional[float] = None
    generation: int = 0
    parent_expressions: List[str] = None
    
    def __post_init__(self):
        if self.parent_expressions is None:
            self.parent_expressions = []

class GeneticMathEngine:
    """Core genetic algorithm for mathematical discovery"""
    
    def __init__(self):
        self.gene_pool: List[MathExpression] = []
        self.generation = 0
        self.constants = Config.TARGET_CONSTANTS
        self.precision = Config.PRECISION_DIGITS
        mpmath.mp.dps = self.precision
        
    def calculate_complexity(self, expression: str) -> int:
        """Calculate expression complexity based on length and operators"""
        # Count operators, functions, and nested structures
        operators = len(re.findall(r'[+\-*/^()]', expression))
        functions = len(re.findall(r'(sqrt|log|exp|sin|cos|tan)', expression))
        nesting = expression.count('(')
        return len(expression) + operators * 2 + functions * 3 + nesting * 2
    
    def evaluate_expression(self, expression: str, target_value: float) -> Tuple[float, float]:
        """Safely evaluate mathematical expression and calculate error"""
        try:
            # Replace mathematical constants
            expr = expression.replace('π', str(mpmath.pi))
            expr = expr.replace('e', str(mpmath.e))
            expr = expr.replace('φ', str((1 + mpmath.sqrt(5))/2))
            expr = expr.replace('γ', str(mpmath.euler))
            
            # Evaluate with high precision
            result = float(mpmath.eval(expr))
            error = abs(result - target_value)
            return result, error
            
        except Exception as e:
            return float('inf'), float('inf')
    
    def calculate_elegance_score(self, expression: MathExpression) -> float:
        """Calculate elegance score: Error × (1 + complexity_penalty × length)"""
        if expression.error is None or expression.error == float('inf'):
            return float('inf')
        
        complexity_factor = 1 + Config.ELEGANCE_COMPLEXITY_PENALTY * expression.complexity
        return expression.error * complexity_factor
    
    def generate_random_expression(self) -> str:
        """Generate a random mathematical expression"""
        templates = [
            # Basic arithmetic combinations
            "π + {a}",
            "π - {a}",
            "π * {a}",
            "π / {a}",
            "{a} / π",
            
            # Exponential forms
            "e^({a})",
            "e^(π/{a})",
            "e^(π*{a})",
            
            # Square roots and radicals
            "sqrt({a})",
            "sqrt(π + {a})",
            "sqrt(π * {a})",
            "sqrt({a} + sqrt({b}))",
            
            # Golden ratio combinations
            "φ^{a}",
            "φ + {a}",
            "φ * {a}",
            
            # Continued fractions
            "{a} + 1/({b} + 1/{c})",
            "π + 1/({a} + 1/{b})",
            
            # Nested expressions
            "({a} + {b}) / ({c} + {d})",
            "sqrt({a}) + sqrt({b})",
            "log({a}) + π",
            
            # Ramanujan-style
            "e^(π * sqrt({a}))",
            "π^2 / {a}",
            "163 + {a}",
            "sqrt(163) + {a}",
        ]
        
        template = random.choice(templates)
        
        # Generate random coefficients
        coeffs = {}
        for var in ['a', 'b', 'c', 'd']:
            if '{' + var + '}' in template:
                coeffs[var] = random.choice([
                    str(random.randint(1, 20)),
                    str(random.randint(1, 200)),
                    str(round(random.uniform(0.1, 10), 3)),
                    'π', 'e', 'φ', '2', '3', '5', '7', '163'
                ])
        
        return template.format(**coeffs)
    
    def mutate_expression(self, expression: str) -> str:
        """Mutate an existing expression"""
        mutations = [
            # Coefficient mutations
            lambda e: re.sub(r'\d+', lambda m: str(int(m.group()) + random.randint(-5, 5)), e),
            
            # Operator mutations
            lambda e: e.replace('+', '-') if '+' in e else e.replace('-', '+'),
            lambda e: e.replace('*', '/') if '*' in e else e.replace('/', '*'),
            
            # Function additions
            lambda e: f"sqrt({e})",
            lambda e: f"log({e})" if not e.startswith('log') else e,
            lambda e: f"exp({e})" if not e.startswith('exp') else e,
            
            # Constant substitutions
            lambda e: e.replace('π', 'e') if 'π' in e else e.replace('e', 'π'),
            lambda e: e.replace('φ', 'π') if 'φ' in e else e,
            
            # Structural changes
            lambda e: f"({e}) + 1",
            lambda e: f"({e}) / 2",
            lambda e: f"2 * ({e})",
        ]
        
        mutation = random.choice(mutations)
        try:
            return mutation(expression)
        except:
            return expression
    
    def crossover_expressions(self, expr1: str, expr2: str) -> str:
        """Combine two expressions"""
        combinations = [
            f"({expr1}) + ({expr2})",
            f"({expr1}) - ({expr2})",
            f"({expr1}) * ({expr2})",
            f"({expr1}) / ({expr2})",
            f"sqrt(({expr1}) * ({expr2}))",
            f"({expr1} + {expr2}) / 2",
        ]
        return random.choice(combinations)
    
    def evaluate_candidate(self, expression: str, target_value: float) -> MathExpression:
        """Create and evaluate a mathematical expression candidate"""
        value, error = self.evaluate_expression(expression, target_value)
        complexity = self.calculate_complexity(expression)
        
        candidate = MathExpression(
            expression=expression,
            value=value,
            error=error,
            complexity=complexity,
            generation=self.generation
        )
        
        candidate.elegance_score = self.calculate_elegance_score(candidate)
        return candidate
    
    def update_gene_pool(self, new_candidates: List[MathExpression]):
        """Update gene pool with new candidates"""
        # Combine with existing gene pool
        all_candidates = self.gene_pool + new_candidates
        
        # Remove duplicates and invalid expressions
        valid_candidates = []
        seen_expressions = set()
        
        for candidate in all_candidates:
            if (candidate.expression not in seen_expressions and 
                candidate.error != float('inf') and 
                candidate.elegance_score != float('inf')):
                valid_candidates.append(candidate)
                seen_expressions.add(candidate.expression)
        
        # Sort by elegance score and keep top candidates
        valid_candidates.sort(key=lambda x: x.elegance_score)
        self.gene_pool = valid_candidates[:Config.GENE_POOL_SIZE]
    
    def get_best_candidates(self, n: int = 10) -> List[MathExpression]:
        """Get the best candidates from gene pool"""
        return sorted(self.gene_pool, key=lambda x: x.elegance_score)[:n]
    
    def generate_population(self, size: int, target_value: float) -> List[MathExpression]:
        """Generate a population of mathematical expressions"""
        population = []
        
        # Generate new random expressions (exploration)
        for _ in range(size // 3):
            expr = self.generate_random_expression()
            candidate = self.evaluate_candidate(expr, target_value)
            population.append(candidate)
        
        # Mutate existing good expressions (exploitation)
        if self.gene_pool:
            for _ in range(size // 3):
                parent = random.choice(self.gene_pool[:10])  # Top 10
                mutated_expr = self.mutate_expression(parent.expression)
                candidate = self.evaluate_candidate(mutated_expr, target_value)
                candidate.parent_expressions = [parent.expression]
                population.append(candidate)
        
        # Crossover between good expressions (recombination)
        if len(self.gene_pool) >= 2:
            for _ in range(size - len(population)):
                parent1, parent2 = random.sample(self.gene_pool[:15], 2)
                crossed_expr = self.crossover_expressions(parent1.expression, parent2.expression)
                candidate = self.evaluate_candidate(crossed_expr, target_value)
                candidate.parent_expressions = [parent1.expression, parent2.expression]
                population.append(candidate)
        
        # Fill remaining with random expressions
        while len(population) < size:
            expr = self.generate_random_expression()
            candidate = self.evaluate_candidate(expr, target_value)
            population.append(candidate)
        
        return population
    
    def evolve_generation(self, target_value: float, population_size: int = 100) -> List[MathExpression]:
        """Evolve one generation"""
        self.generation += 1
        population = self.generate_population(population_size, target_value)
        self.update_gene_pool(population)
        return population
