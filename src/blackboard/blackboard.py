"""Blackboard system for shared memory between agents."""

import logging
import threading
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set
from collections import deque
from pathlib import Path
from enum import Enum


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Blackboard event types."""

    WRITE = "write"
    DELETE = "delete"
    PRUNE = "prune"
    SNAPSHOT = "snapshot"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


@dataclass
class BlackboardEvent:
    """Event emitted by blackboard changes."""

    event_type: EventType
    section: str
    key: Optional[str]
    requester: str
    timestamp: float
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "section": self.section,
            "key": self.key,
            "requester": self.requester,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class SectionConfig:
    """Configuration for a blackboard section."""

    max_tokens: int
    priority: int = 0  # Higher = more important during pruning
    ttl: Optional[float] = None  # Time-to-live in seconds


class Blackboard:
    """
    Shared memory system for agent collaboration.

    Features:
    - Hierarchical sections with token limits
    - Thread-safe read/write operations
    - Event subscription system
    - Auto-pruning when limits exceeded
    - Snapshot/restore capability
    """

    # Default section configurations
    DEFAULT_SECTIONS = {
        "current_goal": SectionConfig(max_tokens=200, priority=10),
        "facts": SectionConfig(max_tokens=500, priority=8),
        "hypotheses": SectionConfig(max_tokens=300, priority=5),
        "artifacts": SectionConfig(max_tokens=5000, priority=7),
        "working_memory": SectionConfig(max_tokens=2000, priority=3),
        "metrics": SectionConfig(max_tokens=500, priority=2),
    }

    def __init__(
        self,
        sections: Optional[Dict[str, SectionConfig]] = None,
        snapshot_interval: float = 10.0,
        max_event_log: int = 50,
    ):
        """
        Initialize blackboard.

        Args:
            sections: Section configurations (uses defaults if None)
            snapshot_interval: Seconds between auto-snapshots
            max_event_log: Maximum events to keep in log
        """
        self.section_configs = sections or self.DEFAULT_SECTIONS.copy()

        # Data storage
        self._data: Dict[str, Dict[str, Any]] = {
            section: {} for section in self.section_configs
        }

        # Token counts per section
        self._token_counts: Dict[str, int] = {
            section: 0 for section in self.section_configs
        }

        # Section locks for thread safety
        self._locks: Dict[str, threading.RLock] = {
            section: threading.RLock() for section in self.section_configs
        }
        self._global_lock = threading.RLock()

        # Event system
        self._subscribers: Dict[EventType, List[Callable[[BlackboardEvent], None]]] = {
            event_type: [] for event_type in EventType
        }
        self._event_log: deque = deque(maxlen=max_event_log)

        # Versioning
        self._version = 0

        # Snapshot management
        self._snapshot_interval = snapshot_interval
        self._last_snapshot_time = time.time()
        self._snapshots: deque = deque(maxlen=10)

        logger.info(f"Blackboard initialized with sections: {list(self.section_configs.keys())}")

    def _estimate_tokens(self, value: Any) -> int:
        """Estimate token count for a value."""
        text = json.dumps(value) if not isinstance(value, str) else value
        return len(text) // 4

    def _emit_event(self, event: BlackboardEvent) -> None:
        """Emit an event to subscribers."""
        self._event_log.append(event)

        for callback in self._subscribers.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def read(self, section: str, requester: str) -> Optional[Dict[str, Any]]:
        """
        Read all data from a section.

        Args:
            section: Section name
            requester: Requesting agent name

        Returns:
            Section data or None if section doesn't exist
        """
        if section not in self._data:
            logger.warning(f"Section '{section}' does not exist")
            return None

        with self._locks[section]:
            return self._data[section].copy()

    def read_key(self, section: str, key: str, requester: str) -> Optional[Any]:
        """
        Read a specific key from a section.

        Args:
            section: Section name
            key: Key to read
            requester: Requesting agent name

        Returns:
            Value or None
        """
        if section not in self._data:
            return None

        with self._locks[section]:
            return self._data[section].get(key)

    def write(
        self,
        section: str,
        key: str,
        value: Any,
        requester: str,
    ) -> bool:
        """
        Write a value to the blackboard.

        Args:
            section: Section name
            key: Key to write
            value: Value to write
            requester: Requesting agent name

        Returns:
            True if write succeeded
        """
        if section not in self._data:
            logger.warning(f"Section '{section}' does not exist")
            return False

        config = self.section_configs[section]
        new_tokens = self._estimate_tokens(value)

        with self._locks[section]:
            # Check token limit
            old_tokens = self._estimate_tokens(self._data[section].get(key, ""))
            net_tokens = new_tokens - old_tokens

            if self._token_counts[section] + net_tokens > config.max_tokens:
                # Try to prune first
                self._prune_section(section)

                # Check again
                if self._token_counts[section] + net_tokens > config.max_tokens:
                    logger.warning(
                        f"Section '{section}' token limit exceeded "
                        f"({self._token_counts[section]} + {net_tokens} > {config.max_tokens})"
                    )
                    return False

            # Write value
            self._data[section][key] = value
            self._token_counts[section] += net_tokens
            self._version += 1

        # Emit event
        event = BlackboardEvent(
            event_type=EventType.WRITE,
            section=section,
            key=key,
            requester=requester,
            timestamp=time.time(),
            data={"tokens": new_tokens},
        )
        self._emit_event(event)

        logger.debug(f"Write: {section}/{key} by {requester} ({new_tokens} tokens)")
        return True

    def delete(self, section: str, key: str, requester: str) -> bool:
        """
        Delete a key from a section.

        Args:
            section: Section name
            key: Key to delete
            requester: Requesting agent name

        Returns:
            True if deleted
        """
        if section not in self._data:
            return False

        with self._locks[section]:
            if key not in self._data[section]:
                return False

            old_tokens = self._estimate_tokens(self._data[section][key])
            del self._data[section][key]
            self._token_counts[section] -= old_tokens
            self._version += 1

        event = BlackboardEvent(
            event_type=EventType.DELETE,
            section=section,
            key=key,
            requester=requester,
            timestamp=time.time(),
        )
        self._emit_event(event)

        logger.debug(f"Delete: {section}/{key} by {requester}")
        return True

    def _prune_section(self, section: str) -> int:
        """
        Prune a section to free space.

        Strategy: Remove oldest entries first.

        Args:
            section: Section to prune

        Returns:
            Number of entries removed
        """
        # Simple strategy: remove entries until under 75% of limit
        config = self.section_configs[section]
        target = int(config.max_tokens * 0.75)
        removed = 0

        keys = list(self._data[section].keys())
        for key in keys:
            if self._token_counts[section] <= target:
                break

            tokens = self._estimate_tokens(self._data[section][key])
            del self._data[section][key]
            self._token_counts[section] -= tokens
            removed += 1

        if removed > 0:
            event = BlackboardEvent(
                event_type=EventType.PRUNE,
                section=section,
                key=None,
                requester="system",
                timestamp=time.time(),
                data={"entries_removed": removed},
            )
            self._emit_event(event)
            logger.info(f"Pruned {removed} entries from section '{section}'")

        return removed

    def prune_all(self) -> Dict[str, int]:
        """
        Prune all sections.

        Returns:
            Dictionary of sections to entries removed
        """
        result = {}
        for section in self._data:
            with self._locks[section]:
                result[section] = self._prune_section(section)
        return result

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[BlackboardEvent], None],
    ) -> None:
        """
        Subscribe to blackboard events.

        Args:
            event_type: Event type to subscribe to
            callback: Callback function
        """
        with self._global_lock:
            self._subscribers[event_type].append(callback)

        event = BlackboardEvent(
            event_type=EventType.SUBSCRIBE,
            section="",
            key=None,
            requester="system",
            timestamp=time.time(),
            data={"subscribed_to": event_type.value},
        )
        self._emit_event(event)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[BlackboardEvent], None],
    ) -> bool:
        """
        Unsubscribe from blackboard events.

        Args:
            event_type: Event type
            callback: Callback to remove

        Returns:
            True if unsubscribed
        """
        with self._global_lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                return True
        return False

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the blackboard.

        Returns:
            Snapshot dictionary
        """
        with self._global_lock:
            snapshot = {
                "version": self._version,
                "timestamp": time.time(),
                "data": {},
                "token_counts": self._token_counts.copy(),
            }

            for section in self._data:
                with self._locks[section]:
                    snapshot["data"][section] = self._data[section].copy()

        self._snapshots.append(snapshot)
        self._last_snapshot_time = time.time()

        event = BlackboardEvent(
            event_type=EventType.SNAPSHOT,
            section="",
            key=None,
            requester="system",
            timestamp=time.time(),
            data={"version": self._version},
        )
        self._emit_event(event)

        logger.debug(f"Snapshot created (version {self._version})")
        return snapshot

    def restore(self, snapshot: Dict[str, Any]) -> bool:
        """
        Restore blackboard from snapshot.

        Args:
            snapshot: Snapshot to restore

        Returns:
            True if restored
        """
        try:
            with self._global_lock:
                for section in self._data:
                    with self._locks[section]:
                        self._data[section] = snapshot["data"].get(section, {}).copy()
                        self._token_counts[section] = snapshot["token_counts"].get(
                            section, 0
                        )

                self._version = snapshot["version"]

            logger.info(f"Restored to snapshot version {self._version}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def save_to_file(self, filepath: Path) -> None:
        """Save blackboard to file."""
        snapshot = self.snapshot()
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)
        logger.info(f"Blackboard saved to {filepath}")

    def load_from_file(self, filepath: Path) -> bool:
        """Load blackboard from file."""
        try:
            with open(filepath, "r") as f:
                snapshot = json.load(f)
            return self.restore(snapshot)
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def get_section_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all sections.

        Returns:
            Statistics dictionary
        """
        stats = {}
        for section, config in self.section_configs.items():
            with self._locks[section]:
                stats[section] = {
                    "entries": len(self._data[section]),
                    "tokens_used": self._token_counts[section],
                    "max_tokens": config.max_tokens,
                    "utilization": self._token_counts[section] / config.max_tokens,
                }
        return stats

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get recent event log."""
        return [event.to_dict() for event in self._event_log]

    @property
    def version(self) -> int:
        """Current version number."""
        return self._version

    def clear(self) -> None:
        """Clear all data from blackboard."""
        with self._global_lock:
            for section in self._data:
                with self._locks[section]:
                    self._data[section] = {}
                    self._token_counts[section] = 0
            self._version += 1
        logger.info("Blackboard cleared")

    def add_section(self, name: str, config: SectionConfig) -> bool:
        """
        Add a new section to the blackboard.

        Args:
            name: Section name
            config: Section configuration

        Returns:
            True if added
        """
        if name in self._data:
            return False

        with self._global_lock:
            self.section_configs[name] = config
            self._data[name] = {}
            self._token_counts[name] = 0
            self._locks[name] = threading.RLock()

        logger.info(f"Added section: {name}")
        return True
