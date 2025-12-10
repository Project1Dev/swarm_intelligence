"""Agent registry for dynamic instantiation and management."""

import logging
from typing import Dict, Type, Optional, List, Any
from pathlib import Path
import yaml

from agents.base import BaseAgent, AgentConfig, AgentRole, Permission


logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for agent types and instances.

    Supports:
    - Registration of agent classes
    - Dynamic instantiation from config
    - Agent lifecycle management
    - Serialization/deserialization
    """

    def __init__(self):
        """Initialize agent registry."""
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        self._instances: Dict[str, BaseAgent] = {}
        self._configs: Dict[str, AgentConfig] = {}

    def register_class(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class.

        Args:
            name: Class name/identifier
            agent_class: Agent class to register
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"{agent_class} is not a subclass of BaseAgent")

        self._agent_classes[name] = agent_class
        logger.info(f"Registered agent class: {name}")

    def unregister_class(self, name: str) -> bool:
        """
        Unregister an agent class.

        Args:
            name: Class name to unregister

        Returns:
            True if unregistered
        """
        if name in self._agent_classes:
            del self._agent_classes[name]
            logger.info(f"Unregistered agent class: {name}")
            return True
        return False

    def get_class(self, name: str) -> Optional[Type[BaseAgent]]:
        """
        Get a registered agent class.

        Args:
            name: Class name

        Returns:
            Agent class or None
        """
        return self._agent_classes.get(name)

    def list_classes(self) -> List[str]:
        """List all registered class names."""
        return list(self._agent_classes.keys())

    def create_agent(
        self,
        class_name: str,
        config: AgentConfig,
        ollama_client: Any,
        blackboard: Optional[Any] = None,
    ) -> BaseAgent:
        """
        Create an agent instance.

        Args:
            class_name: Registered class name
            config: Agent configuration
            ollama_client: Ollama client
            blackboard: Optional blackboard reference

        Returns:
            Created agent instance

        Raises:
            ValueError: If class not registered
        """
        agent_class = self._agent_classes.get(class_name)
        if agent_class is None:
            raise ValueError(f"Agent class '{class_name}' not registered")

        agent = agent_class(config, ollama_client, blackboard)
        self._instances[config.name] = agent
        self._configs[config.name] = config

        logger.info(f"Created agent instance: {config.name} ({class_name})")
        return agent

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get an agent instance by name.

        Args:
            name: Agent name

        Returns:
            Agent instance or None
        """
        return self._instances.get(name)

    def list_agents(self) -> List[str]:
        """List all agent instance names."""
        return list(self._instances.keys())

    def get_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """
        Get all agents with a specific role.

        Args:
            role: Agent role

        Returns:
            List of matching agents
        """
        return [
            agent for agent in self._instances.values()
            if agent.role == role
        ]

    def remove_agent(self, name: str) -> bool:
        """
        Remove an agent instance.

        Args:
            name: Agent name

        Returns:
            True if removed
        """
        if name in self._instances:
            agent = self._instances[name]
            agent.deactivate()
            del self._instances[name]
            if name in self._configs:
                del self._configs[name]
            logger.info(f"Removed agent: {name}")
            return True
        return False

    def activate_all(self) -> None:
        """Activate all agents."""
        for agent in self._instances.values():
            agent.activate()

    def deactivate_all(self) -> None:
        """Deactivate all agents."""
        for agent in self._instances.values():
            agent.deactivate()

    def reset_all_budgets(self) -> None:
        """Reset token budgets for all agents."""
        for agent in self._instances.values():
            agent.reset_budget()

    def save_all_states(self) -> Dict[str, Any]:
        """
        Save all agent states.

        Returns:
            Dictionary of agent states
        """
        return {
            name: agent.save_state()
            for name, agent in self._instances.items()
        }

    def load_agent_state(self, name: str, state: Dict[str, Any]) -> bool:
        """
        Load state for a specific agent.

        Args:
            name: Agent name
            state: State dictionary

        Returns:
            True if loaded
        """
        agent = self._instances.get(name)
        if agent:
            agent.load_state(state)
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all agents.

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_agents": len(self._instances),
            "active_agents": sum(1 for a in self._instances.values() if a.is_active),
            "by_role": {},
            "total_tokens_used": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
        }

        for agent in self._instances.values():
            role = agent.role.value
            if role not in stats["by_role"]:
                stats["by_role"][role] = 0
            stats["by_role"][role] += 1

            stats["total_tokens_used"] += agent.state.tokens_used
            stats["total_tasks_completed"] += agent.state.tasks_completed
            stats["total_tasks_failed"] += agent.state.tasks_failed

        return stats


def load_agent_configs(config_path: Path) -> List[AgentConfig]:
    """
    Load agent configurations from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        List of agent configurations
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    configs = []
    for agent_data in data.get("agents", []):
        # Parse role
        role = AgentRole(agent_data["role"])

        # Parse permissions
        permissions = {}
        for section, perm in agent_data.get("permissions", {}).items():
            permissions[section] = Permission(perm)

        config = AgentConfig(
            name=agent_data["name"],
            role=role,
            model=agent_data["model"],
            temperature=agent_data.get("temperature", 0.7),
            max_tokens=agent_data.get("max_tokens", 1000),
            token_budget=agent_data.get("token_budget", 5000),
            permissions=permissions,
            system_prompt=agent_data.get("system_prompt", ""),
        )
        configs.append(config)

    return configs


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry
