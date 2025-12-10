"""Tests for logging configuration."""

import pytest
import tempfile
import logging
from pathlib import Path

from utils.logging_config import setup_logging, get_logger


class TestLoggingConfig:
    """Test logging configuration."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = setup_logging(log_dir=log_dir)

            assert logger is not None
            assert logging.getLogger().level == logging.INFO

    def test_setup_logging_to_file(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_dir=log_dir, log_to_file=True, log_to_console=False)

            # Check that log file was created
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1

    def test_setup_logging_console_only(self):
        """Test logging to console only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_dir=log_dir, log_to_file=False, log_to_console=True)

            # Check that no log file was created
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 0

    def test_setup_logging_custom_level(self):
        """Test custom logging level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_dir=log_dir, log_level=logging.DEBUG)

            assert logging.getLogger().level == logging.DEBUG

    def test_get_logger(self):
        """Test getting a named logger."""
        logger = get_logger("test_module")

        assert logger is not None
        assert logger.name == "test_module"

    def test_logging_message(self):
        """Test that logging actually works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_dir=log_dir, log_to_file=True, log_to_console=False)

            logger = get_logger("test")
            logger.info("Test message")

            # Check that message was written
            log_files = list(log_dir.glob("*.log"))
            content = log_files[0].read_text()
            assert "Test message" in content
