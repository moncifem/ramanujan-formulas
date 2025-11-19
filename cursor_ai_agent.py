"""
Cursor AI Integration for Ramanujan-Swarm
Uses local AI capabilities instead of external API
"""

import random
import re
import logging
from typing import List, Dict, Any
from genetic_core import GeneticMathEngine

logger = logging.getLogger(__name__)

class CursorMathAgent:
    """Mathematical agent that uses built-in mathematical knowledge instead of external AI"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.genetic_engine = GeneticMathEngine()
        
        # Pre-defined mathematical patterns and templates
        self.expression_templates = {
            'explorer': [
                # Novel creative expressions
                "π^{a} + {b}",
                "e^(π/{a}) - {b}",
                "sqrt({a}*π + {b})",
                "φ^{a} - 1/φ^{b}",
                "log(π^{a} + e^{b})",
                "sin(π/{a}) + cos(π/{b})",
                "γ + π/{a}",
                "ζ(3) * {a} + {b}",
                "sqrt({a} + sqrt({b} + sqrt({c})))",
                "({a} + π)^(1/{b})",
                "e^(π*sqrt({a})) / {b}",
                "163 + e^(-π*sqrt({a}))",
                "π^2/{a} + {b}",
                "φ + 1/(φ + 1/{a})",
                "sqrt(π*e) + {a}",
            ],
            'mutator': [
                # Variations of known expressions
                "π + {delta}",
                "π - {delta}",
                "π * (1 + {delta})",
                "π / (1 + {delta})",
                "e + {delta}",
                "e^(1 + {delta})",
                "φ + {delta}",
                "sqrt(π + {delta})",
                "log(e + {delta})",
                "γ + {delta}",
                "ζ(3) + {delta}",
            ],
            'hybrid': [
                # Combinations and crossovers
                "({expr1}) + ({expr2})",
                "({expr1}) - ({expr2})",
                "({expr1}) * ({expr2})",
                "({expr1}) / ({expr2})",
                "sqrt(({expr1}) * ({expr2}))",
                "log(({expr1}) + ({expr2}))",
                "exp(({expr1}) / ({expr2}))",
            ]
        }
        
        # Mathematical constants and their approximations
        self.constants = {
            'π': 3.141592653589793,
            'e': 2.718281828459045,
            'φ': 1.618033988749895,
            'γ': 0.5772156649015329,
            'ζ(3)': 1.2020569031595942
        }
        
        # Common mathematical coefficients
        self.coefficients = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 25, 30, 32, 36, 40, 48, 50, 60, 64, 72, 80, 90, 96, 100,
            120, 144, 163, 180, 200, 240, 256, 300, 360, 400, 480, 500, 600, 720, 800, 900, 1000,
            0.5, 0.25, 0.125, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.5, 2.5, 3.5, 4.5, 7.5
        ]
    
    def generate_coefficient(self) -> str:
        """Generate a random coefficient"""
        coeff = random.choice(self.coefficients)
        if random.random() < 0.1:  # 10% chance for fraction
            num = random.randint(1, 20)
            den = random.randint(2, 20)
            return f"{num}/{den}"
        return str(coeff)
    
    def generate_expression_explorer(self) -> List[str]:
        """Generate expressions for explorer agent"""
        expressions = []
        templates = self.expression_templates['explorer']
        
        for _ in range(random.randint(5, 10)):
            template = random.choice(templates)
            
            # Fill in coefficients
            expr = template
            for var in ['a', 'b', 'c']:
                if '{' + var + '}' in expr:
                    coeff = self.generate_coefficient()
                    expr = expr.replace('{' + var + '}', coeff)
            
            expressions.append(expr)
        
        return expressions
    
    def generate_expression_mutator(self, gene_pool: List[Dict]) -> List[str]:
        """Generate expressions for mutator agent"""
        expressions = []
        
        if not gene_pool:
            return self.generate_expression_explorer()
        
        # Mutate existing expressions
        for _ in range(random.randint(5, 8)):
            if gene_pool:
                parent = random.choice(gene_pool[:10])  # Top 10
                parent_expr = parent.get('expression', 'π')
                
                # Apply mutations
                mutated = self.mutate_expression(parent_expr)
                expressions.append(mutated)
            
            # Also generate some template-based mutations
            template = random.choice(self.expression_templates['mutator'])
            delta = self.generate_coefficient()
            expr = template.replace('{delta}', delta)
            expressions.append(expr)
        
        return expressions
    
    def generate_expression_hybrid(self, gene_pool: List[Dict]) -> List[str]:
        """Generate expressions for hybrid agent"""
        expressions = []
        
        # Mix of exploration and mutation
        expressions.extend(self.generate_expression_explorer()[:3])
        
        if gene_pool and len(gene_pool) >= 2:
            # Create crossovers
            for _ in range(3):
                expr1_data, expr2_data = random.sample(gene_pool[:15], 2)
                expr1 = expr1_data.get('expression', 'π')
                expr2 = expr2_data.get('expression', 'e')
                
                template = random.choice(self.expression_templates['hybrid'])
                if '{expr1}' in template and '{expr2}' in template:
                    combined = template.replace('{expr1}', expr1).replace('{expr2}', expr2)
                    expressions.append(combined)
        
        return expressions
    
    def mutate_expression(self, expression: str) -> str:
        """Apply mutations to an existing expression"""
        mutations = [
            # Coefficient mutations
            lambda e: self.mutate_coefficients(e),
            # Operator mutations
            lambda e: e.replace('+', '-') if '+' in e else e.replace('-', '+'),
            lambda e: e.replace('*', '/') if '*' in e else e.replace('/', '*'),
            # Function additions
            lambda e: f"sqrt({e})" if not e.startswith('sqrt') else e,
            lambda e: f"log({e})" if not e.startswith('log') else e,
            lambda e: f"exp({e})" if not e.startswith('exp') else e,
            # Constant substitutions
            lambda e: e.replace('π', 'e') if 'π' in e and random.random() < 0.3 else e,
            lambda e: e.replace('e', 'φ') if 'e' in e and random.random() < 0.3 else e,
            # Structural changes
            lambda e: f"({e}) + {self.generate_coefficient()}",
            lambda e: f"({e}) / {self.generate_coefficient()}",
            lambda e: f"{self.generate_coefficient()} * ({e})",
        ]
        
        mutation = random.choice(mutations)
        try:
            return mutation(expression)
        except:
            return expression
    
    def mutate_coefficients(self, expression: str) -> str:
        """Mutate numerical coefficients in expression"""
        # Find numbers and replace them
        def replace_number(match):
            try:
                old_num = float(match.group())
                if random.random() < 0.5:
                    # Small change
                    change = random.uniform(-0.1, 0.1) * old_num
                    new_num = old_num + change
                else:
                    # Replace with random coefficient
                    new_num = random.choice(self.coefficients)
                return str(round(new_num, 6))
            except:
                return match.group()
        
        # Replace standalone numbers (not part of constants like π)
        pattern = r'\b\d+\.?\d*\b'
        return re.sub(pattern, replace_number, expression)
    
    async def generate_expressions(self, target_constant: str, gene_pool: List[Dict]) -> List[str]:
        """Generate mathematical expressions based on agent type"""
        try:
            if self.agent_type == 'explorer':
                expressions = self.generate_expression_explorer()
            elif self.agent_type == 'mutator':
                expressions = self.generate_expression_mutator(gene_pool)
            else:  # hybrid
                expressions = self.generate_expression_hybrid(gene_pool)
            
            # Add some target-specific expressions
            target_specific = self.generate_target_specific_expressions(target_constant)
            expressions.extend(target_specific[:3])
            
            return expressions[:10]  # Limit to 10 expressions
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error: {e}")
            # Fallback to genetic algorithm
            return [self.genetic_engine.generate_random_expression() for _ in range(5)]
    
    def generate_target_specific_expressions(self, target_constant: str) -> List[str]:
        """Generate expressions specific to the target constant"""
        target_expressions = {
            'pi': [
                "4 * (1 - 1/3 + 1/5 - 1/7)",
                "sqrt(6 * (1 + 1/4 + 1/9 + 1/16))",
                "22/7",
                "355/113",
                "sqrt(10)",
                "e^(sqrt(2))",
                "4 * arctan(1)",
                "2 * sqrt(2 + sqrt(2 + sqrt(2)))",
            ],
            'e': [
                "1 + 1 + 1/2 + 1/6 + 1/24",
                "(1 + 1/100)^100",
                "lim((1 + 1/n)^n)",
                "exp(1)",
                "sqrt(2*π*1)",
                "1 + 1 + 1/2! + 1/3! + 1/4!",
            ],
            'phi': [
                "(1 + sqrt(5))/2",
                "1 + 1/(1 + 1/(1 + 1/1))",
                "sqrt(1 + sqrt(1 + sqrt(1)))",
                "2*cos(π/5)",
                "1.618033988749895",
            ],
            'gamma': [
                "0.5772156649015329",
                "lim(1 + 1/2 + 1/3 + ... + 1/n - log(n))",
                "-Γ'(1)",
            ],
            'zeta3': [
                "1 + 1/8 + 1/27 + 1/64",
                "ζ(3)",
                "1.2020569031595943",
            ]
        }
        
        return target_expressions.get(target_constant, ["π", "e", "φ"])

class CursorSwarmOrchestrator:
    """Simplified swarm orchestrator using Cursor AI agents"""
    
    def __init__(self):
        self.agents = []
        self.genetic_engine = GeneticMathEngine()
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize Cursor AI agents"""
        agent_types = ["explorer"] * 8 + ["mutator"] * 8 + ["hybrid"] * 4
        
        for i, agent_type in enumerate(agent_types):
            agent = CursorMathAgent(f"cursor_agent_{i}", agent_type)
            self.agents.append(agent)
    
    async def run_generation(self, target_constant: str, gene_pool: List[Dict]) -> List[Dict]:
        """Run one generation with Cursor AI agents"""
        all_candidates = []
        target_value = self.genetic_engine.constants.get(target_constant, 3.141592653589793)
        
        # Process agents in batches to simulate parallelism
        for agent in self.agents:
            try:
                expressions = await agent.generate_expressions(target_constant, gene_pool)
                
                # Evaluate expressions
                for expr in expressions:
                    try:
                        candidate = self.genetic_engine.evaluate_candidate(expr, target_value)
                        all_candidates.append({
                            'expression': candidate.expression,
                            'value': candidate.value,
                            'error': candidate.error,
                            'complexity': candidate.complexity,
                            'elegance_score': candidate.elegance_score,
                            'generation': candidate.generation,
                            'agent_id': agent.agent_id,
                            'agent_type': agent.agent_type
                        })
                    except Exception as e:
                        logger.warning(f"Failed to evaluate expression '{expr}': {e}")
                        
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} failed: {e}")
        
        return all_candidates
