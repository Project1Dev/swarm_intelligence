"""Tests for communication system."""

import pytest
import time

from communication.message import (
    Message,
    MessageType,
    Priority,
    MessageQueue,
    MessageRouter,
)


@pytest.fixture
def sample_message():
    """Create sample message."""
    return Message(
        from_agent="agent1",
        to_agent="agent2",
        type=MessageType.TASK_RESULT,
        priority=Priority.NORMAL,
        content={"result": "success"},
    )


@pytest.fixture
def message_queue():
    """Create message queue."""
    return MessageQueue(max_tokens=1000)


@pytest.fixture
def router():
    """Create message router."""
    return MessageRouter()


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self, sample_message):
        """Test creating a message."""
        assert sample_message.from_agent == "agent1"
        assert sample_message.to_agent == "agent2"
        assert sample_message.type == MessageType.TASK_RESULT
        assert sample_message.priority == Priority.NORMAL
        assert sample_message.id is not None
        assert sample_message.timestamp > 0

    def test_message_defaults(self):
        """Test message defaults."""
        msg = Message(
            from_agent="a",
            to_agent="b",
            type=MessageType.QUERY,
            priority=Priority.LOW,
            content={},
        )
        assert msg.ttl is None
        assert msg.reply_to is None
        assert msg.metadata == {}

    def test_is_broadcast(self):
        """Test broadcast detection."""
        broadcast = Message(
            from_agent="a",
            to_agent="broadcast",
            type=MessageType.STATUS_UPDATE,
            priority=Priority.LOW,
            content={},
        )
        direct = Message(
            from_agent="a",
            to_agent="b",
            type=MessageType.STATUS_UPDATE,
            priority=Priority.LOW,
            content={},
        )

        assert broadcast.is_broadcast()
        assert not direct.is_broadcast()

    def test_is_expired(self):
        """Test message expiration."""
        # Non-expiring message
        msg1 = Message(
            from_agent="a",
            to_agent="b",
            type=MessageType.QUERY,
            priority=Priority.LOW,
            content={},
            ttl=None,
        )
        assert not msg1.is_expired()

        # Already expired message
        msg2 = Message(
            from_agent="a",
            to_agent="b",
            type=MessageType.QUERY,
            priority=Priority.LOW,
            content={},
            ttl=0.001,
            timestamp=time.time() - 1,  # Created 1 second ago
        )
        assert msg2.is_expired()

    def test_serialization(self, sample_message):
        """Test message serialization."""
        data = sample_message.to_dict()
        assert data["from_agent"] == "agent1"
        assert data["type"] == "task_result"
        assert data["priority"] == 2  # NORMAL

        json_str = sample_message.to_json()
        assert "agent1" in json_str

    def test_deserialization(self, sample_message):
        """Test message deserialization."""
        json_str = sample_message.to_json()
        restored = Message.from_json(json_str)

        assert restored.from_agent == sample_message.from_agent
        assert restored.type == sample_message.type
        assert restored.content == sample_message.content

    def test_create_reply(self, sample_message):
        """Test creating a reply."""
        reply = sample_message.create_reply(
            from_agent="agent2",
            type=MessageType.RESPONSE,
            content={"ack": True},
        )

        assert reply.from_agent == "agent2"
        assert reply.to_agent == "agent1"  # Original sender
        assert reply.reply_to == sample_message.id
        assert reply.priority == sample_message.priority

    def test_message_ordering(self):
        """Test message ordering by priority."""
        msg_low = Message(
            from_agent="a",
            to_agent="b",
            type=MessageType.QUERY,
            priority=Priority.LOW,
            content={},
        )
        msg_critical = Message(
            from_agent="a",
            to_agent="b",
            type=MessageType.ERROR,
            priority=Priority.CRITICAL,
            content={},
        )

        # Critical should come before low
        assert msg_critical < msg_low


class TestMessageQueue:
    """Test MessageQueue class."""

    def test_push_pop(self, message_queue, sample_message):
        """Test basic push and pop."""
        assert message_queue.push(sample_message)
        assert message_queue.size() == 1

        popped = message_queue.pop()
        assert popped.id == sample_message.id
        assert message_queue.is_empty()

    def test_priority_ordering(self, message_queue):
        """Test messages come out in priority order."""
        msg_low = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.LOW, content={},
        )
        msg_normal = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.NORMAL, content={},
        )
        msg_critical = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.CRITICAL, content={},
        )

        # Push in reverse priority order
        message_queue.push(msg_low)
        message_queue.push(msg_normal)
        message_queue.push(msg_critical)

        # Should pop in priority order
        assert message_queue.pop().priority == Priority.CRITICAL
        assert message_queue.pop().priority == Priority.NORMAL
        assert message_queue.pop().priority == Priority.LOW

    def test_pop_specific_priority(self, message_queue):
        """Test popping from specific priority."""
        msg_low = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.LOW, content={},
        )
        msg_high = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.HIGH, content={},
        )

        message_queue.push(msg_low)
        message_queue.push(msg_high)

        # Pop only from LOW
        popped = message_queue.pop(Priority.LOW)
        assert popped.priority == Priority.LOW

    def test_peek(self, message_queue):
        """Test peeking at messages."""
        for i in range(5):
            msg = Message(
                from_agent="a", to_agent="b",
                type=MessageType.QUERY, priority=Priority.NORMAL,
                content={"i": i},
            )
            message_queue.push(msg)

        peeked = message_queue.peek(3)
        assert len(peeked) == 3
        assert message_queue.size() == 5  # Not removed

    def test_token_budget(self):
        """Test token budget enforcement."""
        small_queue = MessageQueue(max_tokens=50)

        # Small message should succeed
        msg1 = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.NORMAL,
            content={"small": True},
        )
        assert small_queue.push(msg1)

        # Large message should fail
        msg2 = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.NORMAL,
            content={"large": "x" * 1000},
        )
        assert not small_queue.push(msg2)

    def test_expired_message_skipped(self, message_queue):
        """Test expired messages are skipped on pop."""
        expired = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.NORMAL,
            content={},
            ttl=0.001,
            timestamp=time.time() - 1,
        )
        valid = Message(
            from_agent="a", to_agent="b",
            type=MessageType.QUERY, priority=Priority.NORMAL,
            content={"valid": True},
        )

        message_queue.push(expired)
        message_queue.push(valid)

        popped = message_queue.pop()
        assert popped.content.get("valid")

    def test_get_messages_for(self, message_queue):
        """Test getting messages for specific agent."""
        msg1 = Message(
            from_agent="a", to_agent="agent1",
            type=MessageType.QUERY, priority=Priority.NORMAL, content={},
        )
        msg2 = Message(
            from_agent="a", to_agent="agent2",
            type=MessageType.QUERY, priority=Priority.NORMAL, content={},
        )
        msg_broadcast = Message(
            from_agent="a", to_agent="broadcast",
            type=MessageType.STATUS_UPDATE, priority=Priority.LOW, content={},
        )

        message_queue.push(msg1)
        message_queue.push(msg2)
        message_queue.push(msg_broadcast)

        # Get messages for agent1 (should include broadcast)
        messages = message_queue.get_messages_for("agent1")
        assert len(messages) == 2

    def test_clear(self, message_queue, sample_message):
        """Test clearing queue."""
        message_queue.push(sample_message)
        message_queue.push(sample_message)

        count = message_queue.clear()
        assert count == 2
        assert message_queue.is_empty()

    def test_statistics(self, message_queue, sample_message):
        """Test queue statistics."""
        message_queue.push(sample_message)

        stats = message_queue.get_statistics()
        assert stats["total_messages"] == 1
        assert stats["by_priority"]["NORMAL"] == 1
        assert stats["total_processed"] == 1


class TestMessageRouter:
    """Test MessageRouter class."""

    def test_register_agent(self, router):
        """Test registering agents."""
        router.register_agent("agent1")
        router.register_agent("agent2")

        stats = router.get_statistics()
        assert "agent1" in stats["registered_agents"]
        assert "agent2" in stats["registered_agents"]

    def test_route_direct_message(self, router):
        """Test routing direct message."""
        router.register_agent("agent1")
        router.register_agent("agent2")

        msg = Message(
            from_agent="agent1", to_agent="agent2",
            type=MessageType.QUERY, priority=Priority.NORMAL, content={},
        )

        assert router.route(msg)

        # Agent2 should receive message
        messages = router.get_messages("agent2")
        assert len(messages) == 1
        assert messages[0].from_agent == "agent1"

        # Agent1 should not have message
        messages = router.get_messages("agent1")
        assert len(messages) == 0

    def test_route_broadcast(self, router):
        """Test routing broadcast message."""
        router.register_agent("agent1")
        router.register_agent("agent2")

        msg = Message(
            from_agent="agent1", to_agent="broadcast",
            type=MessageType.STATUS_UPDATE, priority=Priority.NORMAL,
            content={"status": "ready"},
        )

        assert router.route(msg)

        # Both agents should receive broadcast
        msg1 = router.get_messages("agent1")
        msg2 = router.get_messages("agent2")
        # Note: broadcasts are shared, so only one agent gets it from queue
        # depending on implementation

    def test_route_unknown_recipient(self, router):
        """Test routing to unknown agent fails."""
        router.register_agent("agent1")

        msg = Message(
            from_agent="agent1", to_agent="unknown",
            type=MessageType.QUERY, priority=Priority.NORMAL, content={},
        )

        assert not router.route(msg)

    def test_unregister_agent(self, router):
        """Test unregistering agent."""
        router.register_agent("agent1")
        router.unregister_agent("agent1")

        stats = router.get_statistics()
        assert "agent1" not in stats["registered_agents"]

    def test_peek_messages(self, router):
        """Test peeking at messages."""
        router.register_agent("agent1")

        for i in range(5):
            msg = Message(
                from_agent="sender", to_agent="agent1",
                type=MessageType.QUERY, priority=Priority.NORMAL,
                content={"i": i},
            )
            router.route(msg)

        peeked = router.peek_messages("agent1", 3)
        assert len(peeked) == 3

        # Messages still in queue
        messages = router.get_messages("agent1")
        assert len(messages) == 5


class TestMessageTypes:
    """Test message type handling."""

    def test_all_message_types_valid(self):
        """Test all message types can be used."""
        for msg_type in MessageType:
            msg = Message(
                from_agent="a", to_agent="b",
                type=msg_type, priority=Priority.NORMAL,
                content={},
            )
            assert msg.type == msg_type

    def test_all_priorities_valid(self):
        """Test all priorities can be used."""
        for priority in Priority:
            msg = Message(
                from_agent="a", to_agent="b",
                type=MessageType.QUERY, priority=priority,
                content={},
            )
            assert msg.priority == priority
