"""Base agent class for mathematical expression generation."""

import time
import random
from typing import List
from ramanujan_swarm.config import config
from ramanujan_swarm.constants import format_constant_description
from ramanujan_swarm.math_engine.deduplicator import Expression
from ramanujan_swarm.agents.prompts import get_prompt


class BaseAgent:
    """Base class for all agent types."""

    def __init__(self, agent_id: int, agent_type: str):
        """Initialize agent.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (explorer, mutator, hybrid)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type

        # Initialize LLM based on provider
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on config.

        Returns:
            LangChain LLM instance (ChatAnthropic, ChatGoogleGenerativeAI, ChatBedrock, or ChatOpenAI)
        """
        if config.llm_provider == "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                api_key=config.anthropic_api_key,
            )
        elif config.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                google_api_key=config.gemini_api_key,
            )
        elif config.llm_provider == "bedrock":
            import os
            from langchain_aws import ChatBedrock

            # AWS Bedrock supports bearer token authentication
            # Set the environment variable that boto3 will use
            if config.aws_bearer_token:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = config.aws_bearer_token

            # Initialize ChatBedrock
            bedrock_kwargs = {
                "model_id": config.llm_model,
                "region_name": config.aws_region,
                "model_kwargs": {
                    "temperature": config.llm_temperature,
                    "max_tokens": config.llm_max_tokens,
                },
            }

            # Add access keys if provided (alternative to bearer token)
            if config.aws_access_key_id and config.aws_secret_access_key:
                bedrock_kwargs["aws_access_key_id"] = config.aws_access_key_id
                bedrock_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
                bedrock_kwargs["credentials_profile_name"] = None

            return ChatBedrock(**bedrock_kwargs)
        elif config.llm_provider == "blackbox":
            from langchain_openai import ChatOpenAI

            # Blackbox AI uses OpenAI-compatible API
            return ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                api_key=config.blackbox_api_key,
                base_url="https://api.blackbox.ai/v1",  # Blackbox AI endpoint
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")

    async def generate_expressions(
        self,
        gene_pool: List[Expression],
        generation: int,
        target_constants: List[str],
        num_expressions: int = 5,
    ) -> List[Expression]:
        """Generate mathematical expressions.

        Args:
            gene_pool: Current gene pool of top expressions
            generation: Current generation number
            target_constants: List of target constant names
            num_expressions: Number of expressions to generate

        Returns:
            List of Expression objects
        """
        # Pick a random target constant
        target_constant = random.choice(target_constants)
        constant_desc = format_constant_description(target_constant)

        # Get prompt for this agent type
        prompt = get_prompt(
            agent_type=self.agent_type,
            target_constant=target_constant,
            constant_description=constant_desc,
            gene_pool=gene_pool,
            num_expressions=num_expressions,
        )

        # Call LLM
        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content
        except Exception as e:
            print(f"Agent {self.agent_id} LLM error: {e}")
            return []

        # Parse response into expressions
        expressions = self._parse_response(
            response_text, target_constant, generation
        )

        return expressions

    def _parse_response(
        self, response_text: str, target_constant: str, generation: int
    ) -> List[Expression]:
        """Parse LLM response into Expression objects.

        Args:
            response_text: Raw LLM output
            target_constant: Target constant name
            generation: Current generation

        Returns:
            List of Expression objects
        """
        expressions = []
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines and markdown
            if not line or line.startswith("#") or line.startswith("```"):
                continue

            # Skip explanatory text (heuristic: contains spaces and common words)
            if any(
                word in line.lower()
                for word in ["the", "this", "here", "expression", "formula", "example"]
            ):
                if len(line.split()) > 10:  # Likely explanation
                    continue

            # Create Expression object (will be validated later)
            expr = Expression(
                formula_str=line,
                parsed_expr="",  # Will be filled by validator
                target_constant=target_constant,
                agent_type=self.agent_type,
                generation=generation,
                numeric_value="",
                error=float("inf"),
                timestamp=time.time(),
            )
            expressions.append(expr)

        return expressions
