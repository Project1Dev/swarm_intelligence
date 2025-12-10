"""Message protocol for agent communication."""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum, IntEnum
from collections import deque
import threading
import logging


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between agents."""

    # Task-related
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    TASK_FAILED = "task_failed"

    # Planning
    PLAN_UPDATE = "plan_update"
    PLAN_REQUEST = "plan_request"

    # Code/Solution
    CODE_DRAFT = "code_draft"
    CODE_REVIEW = "code_review"
    SOLUTION_PROPOSAL = "solution_proposal"

    # Verification
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESULT = "validation_result"

    # Queries
    QUERY = "query"
    RESPONSE = "response"

    # Status
    STATUS_UPDATE = "status_update"
    WARNING = "warning"
    ERROR = "error"

    # System
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


class Priority(IntEnum):
    """Message priority levels."""

    CRITICAL = 0  # Immediate processing
    HIGH = 1      # Important, process soon
    NORMAL = 2    # Standard priority
    LOW = 3       # Background/log messages


@dataclass
class Message:
    """
    Message between agents.

    Attributes:
        id: Unique message identifier
        from_agent: Sender agent name
        to_agent: Recipient agent name (or "broadcast")
        type: Message type
        priority: Message priority
        timestamp: Creation timestamp
        content: Message payload
        metadata: Additional metadata
        ttl: Time-to-live in seconds (None = no expiry)
        reply_to: ID of message being replied to
    """

    from_agent: str
    to_agent: str  # "broadcast" for all agents
    type: MessageType
    priority: Priority
    content: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None
    reply_to: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def is_broadcast(self) -> bool:
        """Check if message is a broadcast."""
        return self.to_agent == "broadcast"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "type": self.type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "content": self.content,
            "metadata": self.metadata,
            "ttl": self.ttl,
            "reply_to": self.reply_to,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            type=MessageType(data["type"]),
            priority=Priority(data["priority"]),
            timestamp=data["timestamp"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            ttl=data.get("ttl"),
            reply_to=data.get("reply_to"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def create_reply(
        self,
        from_agent: str,
        type: MessageType,
        content: Dict[str, Any],
        priority: Optional[Priority] = None,
    ) -> "Message":
        """
        Create a reply to this message.

        Args:
            from_agent: Agent sending the reply
            type: Reply message type
            content: Reply content
            priority: Priority (defaults to original)

        Returns:
            Reply message
        """
        return Message(
            from_agent=from_agent,
            to_agent=self.from_agent,
            type=type,
            priority=priority or self.priority,
            content=content,
            reply_to=self.id,
        )

    def __lt__(self, other: "Message") -> bool:
        """Compare by priority, then timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class MessageQueue:
    """
    Priority-based message queue.

    Features:
    - Separate queues per priority level
    - FIFO within each priority
    - Token budget tracking
    - Message expiration
    - Thread-safe operations
    """

    def __init__(self, max_tokens: int = 10000):
        """
        Initialize message queue.

        Args:
            max_tokens: Maximum total tokens across all queues
        """
        self.max_tokens = max_tokens
        self._queues: Dict[Priority, deque] = {
            priority: deque() for priority in Priority
        }
        self._token_count = 0
        self._lock = threading.RLock()
        self._message_count = 0
        self._dropped_count = 0

    def _estimate_tokens(self, message: Message) -> int:
        """Estimate token count for a message."""
        return len(message.to_json()) // 4

    def push(self, message: Message) -> bool:
        """
        Add message to appropriate priority queue.

        Args:
            message: Message to add

        Returns:
            True if added, False if dropped
        """
        tokens = self._estimate_tokens(message)

        with self._lock:
            # Check token budget
            if self._token_count + tokens > self.max_tokens:
                # Try to make room by pruning expired/low priority
                self._prune_expired()

                if self._token_count + tokens > self.max_tokens:
                    # Still over budget, drop message
                    logger.warning(
                        f"Message dropped (over budget): {message.type.value} "
                        f"from {message.from_agent}"
                    )
                    self._dropped_count += 1
                    return False

            # Add to queue
            self._queues[message.priority].append(message)
            self._token_count += tokens
            self._message_count += 1

        logger.debug(
            f"Message queued: {message.type.value} ({message.priority.name}) "
            f"from {message.from_agent} to {message.to_agent}"
        )
        return True

    def pop(self, priority: Optional[Priority] = None) -> Optional[Message]:
        """
        Get next message (highest priority first).

        Args:
            priority: Specific priority to pop from (None = any)

        Returns:
            Message or None if empty
        """
        with self._lock:
            if priority is not None:
                return self._pop_from_queue(priority)

            # Pop from highest priority queue that has messages
            for p in Priority:
                message = self._pop_from_queue(p)
                if message is not None:
                    return message

        return None

    def _pop_from_queue(self, priority: Priority) -> Optional[Message]:
        """Pop from a specific priority queue."""
        queue = self._queues[priority]

        while queue:
            message = queue.popleft()
            tokens = self._estimate_tokens(message)
            self._token_count -= tokens

            # Skip expired messages
            if message.is_expired():
                logger.debug(f"Skipping expired message: {message.id}")
                continue

            return message

        return None

    def peek(self, count: int = 1) -> List[Message]:
        """
        View messages without removing.

        Args:
            count: Number of messages to view

        Returns:
            List of messages
        """
        messages = []

        with self._lock:
            remaining = count
            for p in Priority:
                queue = self._queues[p]
                for message in queue:
                    if remaining <= 0:
                        break
                    if not message.is_expired():
                        messages.append(message)
                        remaining -= 1

        return messages

    def _prune_expired(self) -> int:
        """
        Remove expired messages.

        Returns:
            Number of messages removed
        """
        removed = 0

        for priority in Priority:
            queue = self._queues[priority]
            new_queue = deque()

            for message in queue:
                if message.is_expired():
                    self._token_count -= self._estimate_tokens(message)
                    removed += 1
                else:
                    new_queue.append(message)

            self._queues[priority] = new_queue

        if removed > 0:
            logger.debug(f"Pruned {removed} expired messages")

        return removed

    def get_messages_for(self, agent: str) -> List[Message]:
        """
        Get all messages for a specific agent.

        Args:
            agent: Agent name

        Returns:
            List of messages (removes from queue)
        """
        messages = []

        with self._lock:
            for priority in Priority:
                queue = self._queues[priority]
                remaining = deque()

                for message in queue:
                    if message.to_agent == agent or message.is_broadcast():
                        if not message.is_expired():
                            tokens = self._estimate_tokens(message)
                            self._token_count -= tokens
                            messages.append(message)
                    else:
                        remaining.append(message)

                self._queues[priority] = remaining

        return sorted(messages)  # Sort by priority then timestamp

    def clear(self) -> int:
        """
        Clear all messages.

        Returns:
            Number of messages cleared
        """
        with self._lock:
            count = sum(len(q) for q in self._queues.values())
            for priority in Priority:
                self._queues[priority] = deque()
            self._token_count = 0
            return count

    def size(self, priority: Optional[Priority] = None) -> int:
        """
        Get queue size.

        Args:
            priority: Specific priority (None = total)

        Returns:
            Number of messages
        """
        with self._lock:
            if priority is not None:
                return len(self._queues[priority])
            return sum(len(q) for q in self._queues.values())

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "total_messages": self.size(),
                "by_priority": {
                    p.name: len(self._queues[p]) for p in Priority
                },
                "token_count": self._token_count,
                "max_tokens": self.max_tokens,
                "utilization": self._token_count / self.max_tokens,
                "total_processed": self._message_count,
                "total_dropped": self._dropped_count,
            }


class MessageRouter:
    """
    Routes messages between agents.

    Features:
    - Per-agent mailboxes
    - Broadcast support
    - Message filtering
    """

    def __init__(self, max_tokens_per_agent: int = 5000):
        """
        Initialize router.

        Args:
            max_tokens_per_agent: Max tokens per agent mailbox
        """
        self.max_tokens_per_agent = max_tokens_per_agent
        self._mailboxes: Dict[str, MessageQueue] = {}
        self._broadcast_queue = MessageQueue(max_tokens=10000)
        self._lock = threading.RLock()

    def register_agent(self, agent_name: str) -> None:
        """Register an agent for message routing."""
        with self._lock:
            if agent_name not in self._mailboxes:
                self._mailboxes[agent_name] = MessageQueue(
                    max_tokens=self.max_tokens_per_agent
                )
                logger.info(f"Registered agent mailbox: {agent_name}")

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent."""
        with self._lock:
            if agent_name in self._mailboxes:
                del self._mailboxes[agent_name]
                logger.info(f"Unregistered agent mailbox: {agent_name}")

    def route(self, message: Message) -> bool:
        """
        Route a message to its destination.

        Args:
            message: Message to route

        Returns:
            True if routed successfully
        """
        if message.is_broadcast():
            return self._broadcast_queue.push(message)

        with self._lock:
            if message.to_agent not in self._mailboxes:
                logger.warning(f"Unknown recipient: {message.to_agent}")
                return False

            return self._mailboxes[message.to_agent].push(message)

    def get_messages(self, agent_name: str) -> List[Message]:
        """
        Get all messages for an agent.

        Args:
            agent_name: Agent name

        Returns:
            List of messages
        """
        messages = []

        with self._lock:
            # Get from agent's mailbox
            if agent_name in self._mailboxes:
                while True:
                    msg = self._mailboxes[agent_name].pop()
                    if msg is None:
                        break
                    messages.append(msg)

            # Get broadcasts
            broadcast_msgs = self._broadcast_queue.get_messages_for(agent_name)
            messages.extend(broadcast_msgs)

        return sorted(messages)

    def peek_messages(self, agent_name: str, count: int = 5) -> List[Message]:
        """Peek at agent's messages without removing."""
        with self._lock:
            if agent_name in self._mailboxes:
                return self._mailboxes[agent_name].peek(count)
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        with self._lock:
            return {
                "registered_agents": list(self._mailboxes.keys()),
                "per_agent": {
                    name: queue.get_statistics()
                    for name, queue in self._mailboxes.items()
                },
                "broadcast": self._broadcast_queue.get_statistics(),
            }
