import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import random
from datetime import datetime
from genetic_core import GeneticMathEngine, MathExpression
from config import Config
from cursor_ai_agent import CursorSwarmOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SwarmState:
    """State shared across the swarm"""
    generation: int = 0
    target_constant: str = "pi"
    target_value: float = 3.141592653589793
    best_discoveries: List[Dict] = None
    gene_pool: List[Dict] = None
    total_expressions_evaluated: int = 0
    running: bool = False
    
    def __post_init__(self):
        if self.best_discoveries is None:
            self.best_discoveries = []
        if self.gene_pool is None:
            self.gene_pool = []

class MathematicalAgent:
    """Individual agent in the swarm"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type  # "explorer", "mutator", "hybrid"
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=1.1,
            anthropic_api_key=Config.ANTHROPIC_API_KEY
        )
        self.genetic_engine = GeneticMathEngine()
    
    def get_system_prompt(self, target_constant: str, gene_pool: List[Dict]) -> str:
        """Get system prompt based on agent type"""
        base_prompt = f"""You are a mathematical discovery agent in the Ramanujan-Swarm system.
Your goal is to discover novel mathematical expressions that equal or approximate {target_constant}.

Current target: {target_constant} = {Config.TARGET_CONSTANTS.get(target_constant, 'unknown')}

You should generate creative mathematical expressions using:
- Basic arithmetic: +, -, *, /
- Mathematical constants: π (pi), e, φ (phi/golden ratio), γ (gamma), ζ(3)
- Functions: sqrt(), log(), exp(), sin(), cos(), tan()
- Nested structures and continued fractions
- Integer coefficients and simple fractions

Focus on elegant, concise expressions that might reveal deep mathematical relationships.
Avoid overly complex expressions with many nested operations.

Examples of good expressions:
- e^(π*sqrt(163))
- (1 + sqrt(5))/2
- π^2/6
- sqrt(2 + sqrt(2 + sqrt(2)))
- 163 + 1/(e^π - π)

Generate 5-10 unique mathematical expressions, one per line, without explanations."""

        if self.agent_type == "explorer":
            return base_prompt + "\n\nAs an EXPLORER, focus on completely novel expressions and creative combinations."
        
        elif self.agent_type == "mutator":
            gene_pool_str = "\n".join([f"- {expr['expression']} (error: {expr.get('error', 'unknown')})" 
                                     for expr in gene_pool[:10]])
            return base_prompt + f"""

As a MUTATOR, modify and improve these promising expressions from the gene pool:
{gene_pool_str}

Create variations by:
- Changing coefficients slightly
- Adding/removing terms
- Substituting constants
- Applying functions (sqrt, log, exp)"""
        
        else:  # hybrid
            gene_pool_str = "\n".join([f"- {expr['expression']}" for expr in gene_pool[:5]])
            return base_prompt + f"""

As a HYBRID agent, combine exploration with mutation. Use these top expressions for inspiration:
{gene_pool_str}

Mix novel ideas with improvements to existing promising expressions."""
    
    async def generate_expressions(self, target_constant: str, gene_pool: List[Dict]) -> List[str]:
        """Generate mathematical expressions using LLM"""
        try:
            system_prompt = self.get_system_prompt(target_constant, gene_pool)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate mathematical expressions that approximate {target_constant}:")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse expressions from response
            expressions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Clean up the expression
                    expr = line.replace('=', '').strip()
                    if expr and len(expr) > 1:
                        expressions.append(expr)
            
            return expressions[:10]  # Limit to 10 expressions
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error: {e}")
            # Fallback to genetic algorithm
            return [self.genetic_engine.generate_random_expression() for _ in range(5)]
    
    async def process_generation(self, state: SwarmState) -> List[MathExpression]:
        """Process one generation for this agent"""
        target_value = Config.TARGET_CONSTANTS.get(state.target_constant, state.target_value)
        
        # Generate expressions
        expressions = await self.generate_expressions(state.target_constant, state.gene_pool)
        
        # Evaluate expressions
        candidates = []
        for expr in expressions:
            try:
                candidate = self.genetic_engine.evaluate_candidate(expr, target_value)
                candidate.generation = state.generation
                candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to evaluate expression '{expr}': {e}")
        
        return candidates

class RamanujanSwarm:
    """Main orchestrator for the mathematical discovery swarm"""
    
    def __init__(self):
        self.state = SwarmState()
        self.agents: List[MathematicalAgent] = []
        self.genetic_engine = GeneticMathEngine()
        self.discoveries = []
        self.callbacks = []  # For web interface updates
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize the swarm of agents"""
        agent_types = ["explorer"] * 8 + ["mutator"] * 8 + ["hybrid"] * 4
        
        for i, agent_type in enumerate(agent_types):
            agent = MathematicalAgent(f"agent_{i}", agent_type)
            self.agents.append(agent)
    
    def add_callback(self, callback):
        """Add callback for real-time updates"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, data: Dict):
        """Notify all callbacks of events"""
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def run_generation(self) -> Dict[str, Any]:
        """Run one generation of the swarm"""
        self.state.generation += 1
        logger.info(f"Starting generation {self.state.generation}")
        
        # Notify callbacks
        self._notify_callbacks("generation_start", {
            "generation": self.state.generation,
            "target": self.state.target_constant
        })
        
        # Run all agents in parallel
        tasks = []
        for agent in self.agents:
            task = agent.process_generation(self.state)
            tasks.append(task)
        
        # Collect results
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_candidates = []
        for i, result in enumerate(agent_results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} failed: {result}")
            else:
                all_candidates.extend(result)
        
        # Update genetic engine and gene pool
        self.genetic_engine.generation = self.state.generation
        self.genetic_engine.update_gene_pool(all_candidates)
        
        # Update state
        self.state.gene_pool = [asdict(expr) for expr in self.genetic_engine.gene_pool]
        self.state.total_expressions_evaluated += len(all_candidates)
        
        # Find best discoveries
        best_candidates = self.genetic_engine.get_best_candidates(10)
        discoveries = []
        
        for candidate in best_candidates:
            if candidate.error < Config.ERROR_THRESHOLD_VERIFY:
                discovery = {
                    "expression": candidate.expression,
                    "value": candidate.value,
                    "error": candidate.error,
                    "elegance_score": candidate.elegance_score,
                    "generation": candidate.generation,
                    "timestamp": datetime.now().isoformat()
                }
                discoveries.append(discovery)
                
                # Check if this is a new discovery
                if not any(d["expression"] == candidate.expression for d in self.discoveries):
                    self.discoveries.append(discovery)
                    logger.info(f"NEW DISCOVERY: {candidate.expression} = {candidate.value} (error: {candidate.error})")
        
        # Prepare generation summary
        generation_summary = {
            "generation": self.state.generation,
            "expressions_evaluated": len(all_candidates),
            "total_evaluated": self.state.total_expressions_evaluated,
            "best_candidates": [asdict(c) for c in best_candidates[:5]],
            "new_discoveries": discoveries,
            "gene_pool_size": len(self.state.gene_pool),
            "target_constant": self.state.target_constant,
            "target_value": self.state.target_value
        }
        
        # Notify callbacks
        self._notify_callbacks("generation_complete", generation_summary)
        
        return generation_summary
    
    async def run_discovery_session(self, target_constant: str = "pi", max_generations: int = 50):
        """Run a complete discovery session"""
        self.state.target_constant = target_constant
        self.state.target_value = Config.TARGET_CONSTANTS.get(target_constant, 3.141592653589793)
        self.state.running = True
        
        logger.info(f"Starting discovery session for {target_constant}")
        
        try:
            for generation in range(max_generations):
                if not self.state.running:
                    break
                
                summary = await self.run_generation()
                
                # Check for significant discoveries
                if summary["new_discoveries"]:
                    logger.info(f"Generation {generation + 1}: Found {len(summary['new_discoveries'])} new discoveries!")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Discovery session error: {e}")
        finally:
            self.state.running = False
    
    def stop_discovery(self):
        """Stop the discovery session"""
        self.state.running = False
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state for web interface"""
        return {
            "generation": self.state.generation,
            "target_constant": self.state.target_constant,
            "target_value": self.state.target_value,
            "total_evaluated": self.state.total_expressions_evaluated,
            "running": self.state.running,
            "discoveries": self.discoveries,
            "best_candidates": self.state.gene_pool[:10],
            "gene_pool_size": len(self.state.gene_pool)
        }
