"""Base agent class for the swarm intelligence system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING
from enum import Enum
import time
import json

if TYPE_CHECKING:
    from blackboard.blackboard import Blackboard
    from communication.message import Message


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent role types."""

    LEADER = "leader"
    CODE_SPECIALIST = "code_specialist"
    LOGIC_CHECKER = "logic_checker"
    RETRIEVAL_SPECIALIST = "retrieval_specialist"
    BLACKBOARD_MANAGER = "blackboard_manager"


class Permission(Enum):
    """Blackboard access permissions."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    role: AgentRole
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    token_budget: int = 5000  # Per-round budget
    permissions: Dict[str, Permission] = field(default_factory=dict)
    system_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "token_budget": self.token_budget,
            "permissions": {k: v.value for k, v in self.permissions.items()},
            "system_prompt": self.system_prompt,
        }


@dataclass
class AgentState:
    """Agent state for persistence."""

    tokens_used: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_active: float = 0.0
    context_window: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_used": self.tokens_used,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "last_active": self.last_active,
            "context_window": self.context_window,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create from dictionary."""
        return cls(
            tokens_used=data.get("tokens_used", 0),
            tasks_completed=data.get("tasks_completed", 0),
            tasks_failed=data.get("tasks_failed", 0),
            last_active=data.get("last_active", 0.0),
            context_window=data.get("context_window", []),
            metadata=data.get("metadata", {}),
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the swarm.

    Agents are responsible for:
    - Reading/writing to the blackboard
    - Processing tasks with LLM inference
    - Validating their own outputs
    - Managing token budgets
    """

    def __init__(
        self,
        config: AgentConfig,
        ollama_client: Any,
        blackboard: Optional["Blackboard"] = None,
    ):
        """
        Initialize base agent.

        Args:
            config: Agent configuration
            ollama_client: Ollama client for LLM inference
            blackboard: Shared blackboard (can be set later)
        """
        self.config = config
        self.ollama_client = ollama_client
        self.blackboard = blackboard
        self.state = AgentState()
        self._is_active = False

        logger.info(f"Initialized agent: {config.name} ({config.role.value})")

    @property
    def name(self) -> str:
        """Agent name."""
        return self.config.name

    @property
    def role(self) -> AgentRole:
        """Agent role."""
        return self.config.role

    @property
    def model(self) -> str:
        """Model name."""
        return self.config.model

    @property
    def is_active(self) -> bool:
        """Whether agent is currently active."""
        return self._is_active

    @property
    def tokens_remaining(self) -> int:
        """Remaining token budget."""
        return self.config.token_budget - self.state.tokens_used

    def set_blackboard(self, blackboard: "Blackboard") -> None:
        """Set the blackboard reference."""
        self.blackboard = blackboard

    def has_permission(self, section: str, permission: Permission) -> bool:
        """
        Check if agent has permission for a blackboard section.

        Args:
            section: Blackboard section name
            permission: Required permission

        Returns:
            True if agent has permission
        """
        agent_perm = self.config.permissions.get(section)
        if agent_perm is None:
            return False

        if agent_perm == Permission.READ_WRITE:
            return True

        return agent_perm == permission

    def read_blackboard(self, sections: List[str]) -> Dict[str, Any]:
        """
        Read allowed sections from blackboard.

        Args:
            sections: List of section names to read

        Returns:
            Dictionary with section data
        """
        if self.blackboard is None:
            logger.warning(f"Agent {self.name}: No blackboard connected")
            return {}

        result = {}
        for section in sections:
            if self.has_permission(section, Permission.READ):
                data = self.blackboard.read(section, self.name)
                if data is not None:
                    result[section] = data
            else:
                logger.warning(
                    f"Agent {self.name}: No read permission for section '{section}'"
                )

        return result

    def write_blackboard(self, section: str, key: str, value: Any) -> bool:
        """
        Write to allowed section of blackboard.

        Args:
            section: Blackboard section name
            key: Key within section
            value: Value to write

        Returns:
            True if write succeeded
        """
        if self.blackboard is None:
            logger.warning(f"Agent {self.name}: No blackboard connected")
            return False

        if not self.has_permission(section, Permission.WRITE):
            logger.warning(
                f"Agent {self.name}: No write permission for section '{section}'"
            )
            return False

        return self.blackboard.write(section, key, value, self.name)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Simple approximation: ~4 characters per token.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def check_budget(self, estimated_tokens: int) -> bool:
        """
        Check if operation fits within token budget.

        Args:
            estimated_tokens: Estimated tokens for operation

        Returns:
            True if within budget
        """
        return self.state.tokens_used + estimated_tokens <= self.config.token_budget

    def use_tokens(self, tokens: int) -> None:
        """
        Record token usage.

        Args:
            tokens: Tokens used
        """
        self.state.tokens_used += tokens
        logger.debug(
            f"Agent {self.name}: Used {tokens} tokens, "
            f"remaining: {self.tokens_remaining}"
        )

    def reset_budget(self) -> None:
        """Reset token budget for new round."""
        self.state.tokens_used = 0
        logger.debug(f"Agent {self.name}: Budget reset")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, int]:
        """
        Generate text using LLM.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Tuple of (generated text, tokens used)
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temperature is None:
            temperature = self.config.temperature

        # Check budget
        estimated = self.estimate_tokens(prompt) + max_tokens
        if not self.check_budget(estimated):
            logger.warning(f"Agent {self.name}: Insufficient token budget")
            return "", 0

        # Build full prompt with system prompt
        full_prompt = prompt
        if self.config.system_prompt:
            full_prompt = f"{self.config.system_prompt}\n\n{prompt}"

        # Generate
        response = self.ollama_client.generate(
            model=self.model,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response.success:
            self.use_tokens(response.tokens_used)
            return response.content, response.tokens_used
        else:
            logger.error(f"Agent {self.name}: Generation failed: {response.error}")
            return "", 0

    @abstractmethod
    def process_task(self, task: Any, context: Dict[str, Any]) -> Optional["Message"]:
        """
        Process a task and return a message.

        Args:
            task: Task to process
            context: Additional context

        Returns:
            Message with results, or None if failed
        """
        pass

    @abstractmethod
    def validate_output(self, output: Any) -> Tuple[bool, str]:
        """
        Validate the agent's output.

        Args:
            output: Output to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        pass

    def activate(self) -> None:
        """Activate the agent."""
        self._is_active = True
        self.state.last_active = time.time()
        logger.info(f"Agent {self.name}: Activated")

    def deactivate(self) -> None:
        """Deactivate the agent."""
        self._is_active = False
        logger.info(f"Agent {self.name}: Deactivated")

    def save_state(self) -> Dict[str, Any]:
        """
        Save agent state for persistence.

        Returns:
            State dictionary
        """
        return {
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        """
        Load agent state from persistence.

        Args:
            data: State dictionary
        """
        if "state" in data:
            self.state = AgentState.from_dict(data["state"])
        logger.info(f"Agent {self.name}: State loaded")

    def add_to_context(self, entry: Dict[str, Any], max_entries: int = 10) -> None:
        """
        Add entry to context window.

        Args:
            entry: Context entry
            max_entries: Maximum entries to keep
        """
        self.state.context_window.append(entry)
        if len(self.state.context_window) > max_entries:
            self.state.context_window = self.state.context_window[-max_entries:]

    def get_context(self) -> List[Dict[str, Any]]:
        """Get current context window."""
        return self.state.context_window.copy()

    def clear_context(self) -> None:
        """Clear context window."""
        self.state.context_window = []

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"role={self.role.value}, "
            f"model={self.model})"
        )
