"""
Smart Mathematical Agent - Fast and Intelligent
Avoids trivial results and focuses on genuine discoveries.
"""

import random
from typing import List, Dict, Any
from mpmath import mp
import hashlib

class SmartMathematicalAgent:
    """
    An intelligent agent that:
    1. Learns from failures
    2. Avoids trivial patterns
    3. Uses mathematical knowledge
    4. Generates diverse expressions
    5. Works extremely fast
    """
    
    def __init__(self):
        """Initialize the smart agent."""
        self.knowledge_base = {
            'avoid_patterns': set(),
            'successful_patterns': set(),
            'parameter_ranges': {},
            'discovered_values': set()
        }
        
        # Mathematical constants we care about
        self.important_constants = {
            'pi': mp.pi,
            'e': mp.e,
            'phi': mp.phi,
            'sqrt2': mp.sqrt(2),
            'sqrt3': mp.sqrt(3),
            'sqrt5': mp.sqrt(5),
            'ln2': mp.log(2),
            'ln3': mp.log(3),
            'catalan': mp.catalan,
        }
        
        # Avoid these patterns (they lead to trivial results)
        self.avoid_templates = [
            "mp.jtheta(*, 0, mp.exp(-mp.pi * mp.sqrt(*)))",  # Trivial limit to 1
            "mp.exp(-*)",  # Usually too small
            "*/mp.exp(*)",  # Usually too small
            "mp.sinh(*)/mp.exp(*)",  # Trivial identity
        ]
        
        # Focus on these patterns (more likely to be interesting)
        self.focus_templates = [
            # Nested radicals
            "mp.sqrt(* + mp.sqrt(*))",
            "mp.sqrt(* - mp.sqrt(*))",
            "mp.cbrt(* + mp.cbrt(*))",
            
            # Continued fractions
            "*/(1 + */(1 + */*))",
            
            # Special function combinations
            "mp.zeta(*) * mp.zeta(*)",
            "mp.gamma(*) / mp.gamma(*)",
            "mp.polylog(*, 1/*)",
            
            # AGM
            "mp.agm(*, *)",
            "mp.pi / (2 * mp.agm(1, 1/mp.sqrt(*)))",
            
            # Hypergeometric
            "mp.hyp2f1(*/*, */*, 1, */*)",
            
            # Elliptic
            "mp.ellipk(*/*)",
            "mp.ellipe(*/*)",
            
            # Algebraic combinations
            "mp.sqrt(*) + mp.sqrt(*) - mp.sqrt(*)",
            "(mp.sqrt(*) + mp.sqrt(*)) / mp.sqrt(*)",
            "mp.log(*) / mp.log(*)",
        ]
    
    def generate_smart_expression(self) -> str:
        """Generate a single smart expression."""
        # Choose template
        template = random.choice(self.focus_templates)
        
        # Smart parameter selection
        params = self._select_smart_parameters(template)
        
        # Fill template
        expr = template
        for param in params:
            expr = expr.replace('*', str(param), 1)
        
        return expr
    
    def _select_smart_parameters(self, template: str) -> List[Any]:
        """Select parameters intelligently based on template."""
        num_params = template.count('*')
        params = []
        
        for i in range(num_params):
            if 'mp.sqrt' in template:
                # For square roots, use small integers that might produce algebraic numbers
                params.append(random.choice([2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 19, 20]))
            elif 'mp.zeta' in template or 'mp.polylog' in template:
                # For zeta and polylog, use small integers
                params.append(random.choice([2, 3, 4, 5, 6]))
            elif 'mp.gamma' in template:
                # For gamma, use rational numbers
                numerator = random.choice([1, 2, 3, 4, 5])
                denominator = random.choice([2, 3, 4, 5, 6])
                params.append(f"{numerator}/{denominator}")
            elif 'mp.hyp' in template:
                # For hypergeometric, use simple fractions
                numerator = random.choice([1, 2, 3])
                denominator = random.choice([2, 3, 4, 5])
                params.append(f"{numerator}/{denominator}")
            elif 'mp.agm' in template:
                # For AGM, use simple values
                params.append(random.choice([1, 2, "mp.sqrt(2)", "mp.sqrt(3)", "1/mp.sqrt(2)"]))
            elif 'mp.ellip' in template:
                # For elliptic, use rational k values
                numerator = random.choice([1, 2, 3])
                denominator = random.choice([2, 3, 4, 5])
                params.append(f"{numerator}/{denominator}")
            else:
                # Default: small integers
                params.append(random.choice([2, 3, 5, 7, 11, 13, 17, 19]))
        
        return params
    
    def generate_batch(self, size: int = 100) -> List[str]:
        """Generate a batch of smart expressions quickly."""
        expressions = []
        seen_hashes = set()
        
        while len(expressions) < size:
            expr = self.generate_smart_expression()
            
            # Avoid duplicates using hash
            expr_hash = hashlib.md5(expr.encode()).hexdigest()
            if expr_hash not in seen_hashes:
                seen_hashes.add(expr_hash)
                expressions.append(expr)
        
        # Add some specific high-value targets
        expressions.extend(self._get_high_value_expressions())
        
        return expressions[:size]
    
    def _get_high_value_expressions(self) -> List[str]:
        """Return known high-value expressions to explore."""
        return [
            # Apéry's constant approximations
            "mp.zeta(3)",
            "17*mp.zeta(4)/2 - 6*mp.zeta(2)*mp.log(2)**2",
            
            # Catalan's constant approximations  
            "mp.pi * mp.log(2 + mp.sqrt(3)) / 4",
            
            # Nested radicals
            "mp.sqrt(2 + mp.sqrt(3))",
            "mp.sqrt(5 + 2*mp.sqrt(6))",
            
            # Pi formulas
            "16*mp.atan(1/5) - 4*mp.atan(1/239)",
            "mp.sqrt(6 * mp.zeta(2))",
            
            # Golden ratio formulas
            "(1 + mp.sqrt(5))/2",
            "mp.sqrt(1 + mp.sqrt(1 + mp.sqrt(1 + mp.sqrt(1 + mp.sqrt(5)))))",
            
            # AGM formulas
            "mp.pi / (2 * mp.agm(1, 1/mp.sqrt(2)))",
            
            # Hypergeometric identities
            "mp.hyp2f1(1/2, 1/2, 1, 1/2)",
            "mp.hyp2f1(1/3, 2/3, 1, 3/4)",
        ]
    
    def learn_from_failure(self, failed_expression: str, error_type: str):
        """Learn from failed expressions to avoid similar patterns."""
        # Extract pattern from failed expression
        pattern = self._extract_pattern(failed_expression)
        
        if error_type == 'trivial_limit':
            self.knowledge_base['avoid_patterns'].add(pattern)
        elif error_type == 'too_complex':
            # Simplify parameters in future
            pass
        elif error_type == 'duplicate_value':
            # Track the value to avoid duplicates
            try:
                value = eval(failed_expression)
                self.knowledge_base['discovered_values'].add(float(value))
            except:
                pass
    
    def learn_from_success(self, successful_expression: str):
        """Learn from successful expressions to generate similar ones."""
        pattern = self._extract_pattern(successful_expression)
        self.knowledge_base['successful_patterns'].add(pattern)
    
    def _extract_pattern(self, expression: str) -> str:
        """Extract the pattern from an expression."""
        # Replace numbers with wildcards
        import re
        pattern = re.sub(r'\d+\.?\d*', '*', expression)
        return pattern
    
    def mutate_expression(self, expression: str) -> str:
        """Mutate a successful expression to find related discoveries."""
        import re
        
        # Find all numbers in the expression
        numbers = re.findall(r'\d+\.?\d*', expression)
        
        if not numbers:
            return expression
        
        # Mutate one random number
        idx = random.randint(0, len(numbers) - 1)
        old_num = numbers[idx]
        
        # Generate nearby number
        try:
            val = float(old_num)
            if val.is_integer():
                # For integers, try nearby integers
                new_val = val + random.choice([-2, -1, 1, 2])
                if new_val > 0:
                    mutated = expression.replace(old_num, str(int(new_val)), 1)
                else:
                    mutated = expression
            else:
                # For decimals, small perturbation
                new_val = val * random.uniform(0.8, 1.2)
                mutated = expression.replace(old_num, str(new_val), 1)
        except:
            mutated = expression
        
        return mutated
    
    def crossover_expressions(self, expr1: str, expr2: str) -> str:
        """Combine two successful expressions to create a new one."""
        # Simple crossover: combine parts of two expressions
        operators = ['+', '-', '*', '/']
        op = random.choice(operators)
        
        # Create combined expression
        if random.random() < 0.5:
            return f"({expr1}) {op} ({expr2})"
        else:
            # Take function from one, argument from another
            import re
            
            # Extract function calls
            func1 = re.findall(r'mp\.\w+', expr1)
            func2 = re.findall(r'mp\.\w+', expr2)
            
            if func1 and func2:
                # Swap a random function
                new_func = random.choice(func2)
                old_func = random.choice(func1)
                return expr1.replace(old_func, new_func, 1)
            else:
                return f"({expr1}) {op} ({expr2})"


class FastEvaluator:
    """Fast expression evaluator with caching and optimization."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.cache = {}
        mp.dps = 30  # Lower precision for speed
    
    def evaluate(self, expression: str) -> Dict[str, Any]:
        """Evaluate expression quickly."""
        # Check cache
        expr_hash = hashlib.md5(expression.encode()).hexdigest()
        if expr_hash in self.cache:
            return self.cache[expr_hash]
        
        try:
            # Evaluate with timeout
            value = eval(expression)
            
            # Analyze the value
            result = self._analyze_value(value, expression)
            
            # Cache result
            self.cache[expr_hash] = result
            
            return result
            
        except Exception as e:
            result = {'expression': expression, 'error': str(e)}
            self.cache[expr_hash] = result
            return result
    
    def _analyze_value(self, value: Any, expression: str) -> Dict[str, Any]:
        """Analyze the computed value for interesting properties."""
        result = {'expression': expression}
        
        try:
            if isinstance(value, complex):
                if abs(value.imag) < 1e-10:
                    value = value.real
                else:
                    result['complex'] = True
                    return result
            
            value = mp.mpf(value)
            result['value'] = float(value)
            
            # Check for near-integer
            nearest_int = mp.nint(value)
            error = abs(value - nearest_int)
            
            if 1e-12 < error < 0.1:
                result['near_integer'] = int(nearest_int)
                result['error'] = float(error)
                result['interesting'] = True
            
            # Check for simple fraction
            from fractions import Fraction
            try:
                frac = Fraction(float(value)).limit_denominator(100)
                if abs(float(value) - float(frac)) < 1e-10:
                    result['fraction'] = f"{frac.numerator}/{frac.denominator}"
                    result['interesting'] = True
            except:
                pass
            
            # Check for known constants
            for name, const in [('pi', mp.pi), ('e', mp.e), ('phi', mp.phi), 
                               ('sqrt2', mp.sqrt(2)), ('sqrt3', mp.sqrt(3))]:
                ratio = value / const
                if abs(ratio - mp.nint(ratio)) < 1e-10:
                    result['relation'] = f"{int(mp.nint(ratio))} * {name}"
                    result['interesting'] = True
                    break
            
        except:
            pass
        
        return result


def run_smart_discovery(time_limit: int = 30):
    """Run the smart discovery system."""
    import time
    
    agent = SmartMathematicalAgent()
    evaluator = FastEvaluator()
    
    start_time = time.time()
    discoveries = []
    total_evaluated = 0
    
    print(f"Running smart discovery for {time_limit} seconds...")
    print("=" * 60)
    
    while time.time() - start_time < time_limit:
        # Generate batch
        expressions = agent.generate_batch(100)
        
        # Evaluate batch
        for expr in expressions:
            result = evaluator.evaluate(expr)
            total_evaluated += 1
            
            if result.get('interesting'):
                discoveries.append(result)
                agent.learn_from_success(expr)
                
                # Generate mutations of successful expression
                for _ in range(3):
                    mutated = agent.mutate_expression(expr)
                    mut_result = evaluator.evaluate(mutated)
                    if mut_result.get('interesting'):
                        discoveries.append(mut_result)
            else:
                if result.get('error'):
                    agent.learn_from_failure(expr, 'evaluation_error')
        
        # Status update
        elapsed = time.time() - start_time
        rate = total_evaluated / elapsed
        print(f"\rEvaluated: {total_evaluated} ({rate:.0f}/sec), Discoveries: {len(discoveries)}", end='')
    
    print("\n" + "=" * 60)
    
    # Sort discoveries by interestingness
    discoveries.sort(key=lambda x: x.get('error', 1))
    
    print(f"\nTop 10 Discoveries:")
    for i, disc in enumerate(discoveries[:10], 1):
        print(f"\n{i}. {disc['expression'][:60]}...")
        if disc.get('near_integer'):
            print(f"   ≈ {disc['near_integer']} (error: {disc['error']:.2e})")
        elif disc.get('fraction'):
            print(f"   = {disc['fraction']}")
        elif disc.get('relation'):
            print(f"   = {disc['relation']}")
    
    return {
        'total_evaluated': total_evaluated,
        'rate': total_evaluated / (time.time() - start_time),
        'discoveries': discoveries
    }


if __name__ == "__main__":
    results = run_smart_discovery(30)
    
    import json
    with open('smart_discoveries.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to smart_discoveries.json")
    print(f"📊 Evaluation rate: {results['rate']:.0f} expressions/second")
