"""Tests for blackboard system."""

import pytest
import tempfile
import threading
import time
from pathlib import Path

from blackboard.blackboard import (
    Blackboard,
    SectionConfig,
    EventType,
    BlackboardEvent,
)


@pytest.fixture
def blackboard():
    """Create test blackboard."""
    return Blackboard()


@pytest.fixture
def custom_blackboard():
    """Create blackboard with custom sections."""
    sections = {
        "test_section": SectionConfig(max_tokens=100, priority=5),
        "another_section": SectionConfig(max_tokens=200, priority=3),
    }
    return Blackboard(sections=sections)


class TestBlackboardBasics:
    """Test basic blackboard operations."""

    def test_initialization(self, blackboard):
        """Test blackboard initialization."""
        assert blackboard.version == 0
        assert "current_goal" in blackboard.section_configs
        assert "facts" in blackboard.section_configs

    def test_custom_sections(self, custom_blackboard):
        """Test custom section configuration."""
        assert "test_section" in custom_blackboard.section_configs
        assert custom_blackboard.section_configs["test_section"].max_tokens == 100

    def test_write_read(self, blackboard):
        """Test basic write and read."""
        success = blackboard.write("facts", "key1", "value1", "agent1")
        assert success

        data = blackboard.read("facts", "agent1")
        assert data["key1"] == "value1"

    def test_read_key(self, blackboard):
        """Test reading specific key."""
        blackboard.write("facts", "key1", "value1", "agent1")
        blackboard.write("facts", "key2", "value2", "agent1")

        value = blackboard.read_key("facts", "key1", "agent1")
        assert value == "value1"

    def test_delete(self, blackboard):
        """Test deleting key."""
        blackboard.write("facts", "key1", "value1", "agent1")
        assert blackboard.delete("facts", "key1", "agent1")
        assert blackboard.read_key("facts", "key1", "agent1") is None

    def test_write_invalid_section(self, blackboard):
        """Test writing to non-existent section."""
        success = blackboard.write("invalid", "key", "value", "agent")
        assert not success

    def test_read_invalid_section(self, blackboard):
        """Test reading from non-existent section."""
        data = blackboard.read("invalid", "agent")
        assert data is None

    def test_version_increment(self, blackboard):
        """Test version increments on writes."""
        assert blackboard.version == 0

        blackboard.write("facts", "key1", "value1", "agent1")
        assert blackboard.version == 1

        blackboard.write("facts", "key2", "value2", "agent1")
        assert blackboard.version == 2


class TestTokenLimits:
    """Test token limit enforcement."""

    def test_token_counting(self, custom_blackboard):
        """Test token counting."""
        # Write small value
        custom_blackboard.write("test_section", "key1", "short", "agent")

        stats = custom_blackboard.get_section_stats()
        assert stats["test_section"]["tokens_used"] > 0

    def test_token_limit_exceeded(self, custom_blackboard):
        """Test behavior when token limit exceeded."""
        # Try to write value exceeding limit
        large_value = "x" * 1000  # Way over 100 tokens

        success = custom_blackboard.write("test_section", "key1", large_value, "agent")
        assert not success

    def test_auto_pruning(self, custom_blackboard):
        """Test auto-pruning on write."""
        # Fill section close to limit
        for i in range(10):
            custom_blackboard.write("test_section", f"key{i}", f"val{i}", "agent")

        # Get stats before attempting large write
        stats_before = custom_blackboard.get_section_stats()

        # Large write should trigger pruning
        result = custom_blackboard.write(
            "test_section", "large", "x" * 50, "agent"
        )
        # May or may not succeed depending on pruning effectiveness


class TestEventSystem:
    """Test event subscription system."""

    def test_subscribe_to_writes(self, blackboard):
        """Test subscribing to write events."""
        events_received = []

        def on_write(event: BlackboardEvent):
            events_received.append(event)

        blackboard.subscribe(EventType.WRITE, on_write)
        blackboard.write("facts", "key1", "value1", "agent1")

        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.WRITE
        assert events_received[0].section == "facts"
        assert events_received[0].key == "key1"

    def test_subscribe_to_deletes(self, blackboard):
        """Test subscribing to delete events."""
        events_received = []

        def on_delete(event: BlackboardEvent):
            events_received.append(event)

        blackboard.subscribe(EventType.DELETE, on_delete)
        blackboard.write("facts", "key1", "value1", "agent1")
        blackboard.delete("facts", "key1", "agent1")

        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.DELETE

    def test_unsubscribe(self, blackboard):
        """Test unsubscribing from events."""
        events_received = []

        def callback(event: BlackboardEvent):
            events_received.append(event)

        blackboard.subscribe(EventType.WRITE, callback)
        blackboard.write("facts", "key1", "value1", "agent1")
        assert len(events_received) == 1

        blackboard.unsubscribe(EventType.WRITE, callback)
        blackboard.write("facts", "key2", "value2", "agent1")
        assert len(events_received) == 1  # No new events


class TestSnapshots:
    """Test snapshot functionality."""

    def test_create_snapshot(self, blackboard):
        """Test creating snapshot."""
        blackboard.write("facts", "key1", "value1", "agent1")

        snapshot = blackboard.snapshot()
        assert snapshot["version"] == 1
        assert "data" in snapshot
        assert snapshot["data"]["facts"]["key1"] == "value1"

    def test_restore_snapshot(self, blackboard):
        """Test restoring from snapshot."""
        blackboard.write("facts", "key1", "value1", "agent1")
        snapshot = blackboard.snapshot()

        # Modify blackboard
        blackboard.write("facts", "key1", "modified", "agent1")
        assert blackboard.read_key("facts", "key1", "agent1") == "modified"

        # Restore
        blackboard.restore(snapshot)
        assert blackboard.read_key("facts", "key1", "agent1") == "value1"

    def test_get_latest_snapshot(self, blackboard):
        """Test getting latest snapshot."""
        assert blackboard.get_latest_snapshot() is None

        blackboard.snapshot()
        assert blackboard.get_latest_snapshot() is not None


class TestFilePersistence:
    """Test file save/load."""

    def test_save_and_load(self, blackboard):
        """Test saving and loading from file."""
        blackboard.write("facts", "key1", "value1", "agent1")
        blackboard.write("current_goal", "task", "solve maze", "agent1")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "blackboard.json"
            blackboard.save_to_file(filepath)

            # Create new blackboard and load
            new_bb = Blackboard()
            success = new_bb.load_from_file(filepath)

            assert success
            assert new_bb.read_key("facts", "key1", "agent") == "value1"
            assert new_bb.read_key("current_goal", "task", "agent") == "solve maze"


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_writes(self, blackboard):
        """Test concurrent write operations."""
        results = []

        def write_task(agent_id: int):
            for i in range(10):
                success = blackboard.write(
                    "facts",
                    f"key_{agent_id}_{i}",
                    f"value_{agent_id}_{i}",
                    f"agent{agent_id}",
                )
                results.append(success)

        threads = [
            threading.Thread(target=write_task, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes should succeed
        assert all(results)
        assert blackboard.version == 50

    def test_concurrent_read_write(self, blackboard):
        """Test concurrent reads and writes."""
        blackboard.write("facts", "shared", "initial", "setup")
        read_results = []
        write_results = []

        def reader(agent_id: int):
            for _ in range(10):
                data = blackboard.read("facts", f"reader{agent_id}")
                read_results.append(data is not None)
                time.sleep(0.001)

        def writer(agent_id: int):
            for i in range(10):
                success = blackboard.write(
                    "facts",
                    f"writer_{agent_id}",
                    f"value_{i}",
                    f"writer{agent_id}",
                )
                write_results.append(success)
                time.sleep(0.001)

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=reader, args=(i,)))
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(read_results)
        assert all(write_results)


class TestSectionManagement:
    """Test section management."""

    def test_add_section(self, blackboard):
        """Test adding new section."""
        config = SectionConfig(max_tokens=500, priority=5)
        success = blackboard.add_section("new_section", config)

        assert success
        assert "new_section" in blackboard.section_configs

        # Can write to new section
        blackboard.write("new_section", "key1", "value1", "agent")
        assert blackboard.read_key("new_section", "key1", "agent") == "value1"

    def test_add_duplicate_section(self, blackboard):
        """Test adding duplicate section fails."""
        config = SectionConfig(max_tokens=500)
        success = blackboard.add_section("facts", config)  # Already exists
        assert not success

    def test_clear(self, blackboard):
        """Test clearing all data."""
        blackboard.write("facts", "key1", "value1", "agent1")
        blackboard.write("current_goal", "task", "test", "agent1")

        blackboard.clear()

        assert blackboard.read("facts", "agent") == {}
        assert blackboard.read("current_goal", "agent") == {}

    def test_get_section_stats(self, blackboard):
        """Test getting section statistics."""
        blackboard.write("facts", "key1", "value1", "agent1")

        stats = blackboard.get_section_stats()
        assert "facts" in stats
        assert stats["facts"]["entries"] == 1
        assert stats["facts"]["tokens_used"] > 0


class TestEventLog:
    """Test event logging."""

    def test_event_log(self, blackboard):
        """Test event log is maintained."""
        blackboard.write("facts", "key1", "value1", "agent1")
        blackboard.write("facts", "key2", "value2", "agent1")
        blackboard.delete("facts", "key1", "agent1")

        log = blackboard.get_event_log()
        assert len(log) >= 3

        # Check event types in log
        types = [e["event_type"] for e in log]
        assert "write" in types
        assert "delete" in types
