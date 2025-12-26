"""
CAMformer Output System

Logging and output management.
"""

from enum import IntEnum
from typing import Optional
import sys


class OutputLevel(IntEnum):
    """Output verbosity levels"""
    FATAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    VERBOSE = 4
    DEBUG = 5


class Output:
    """
    Output manager for simulation logging.

    Supports multiple verbosity levels and component-specific prefixes.
    """

    # Class-level default settings
    DEFAULT_LEVEL = OutputLevel.INFO
    GLOBAL_ENABLED = True

    def __init__(self, prefix: str = "", level: int = 3, mask: int = 0):
        """
        Initialize output manager.

        Args:
            prefix: Prefix for log messages (usually component name)
            level: Verbosity level (0=FATAL to 5=DEBUG)
            mask: Output mask (reserved for future use)
        """
        self.prefix = prefix
        self.level = OutputLevel(level) if isinstance(level, int) else level
        self.mask = mask
        self.enabled = True

    def set_level(self, level: OutputLevel) -> None:
        """Set output level"""
        self.level = level

    def set_prefix(self, prefix: str) -> None:
        """Set output prefix"""
        self.prefix = prefix

    def _format_message(self, level_name: str, message: str) -> str:
        """Format log message with prefix"""
        if self.prefix:
            return f"[{self.prefix}] {level_name}: {message}"
        return f"{level_name}: {message}"

    def _should_output(self, level: OutputLevel) -> bool:
        """Check if message should be output"""
        return (self.enabled and
                Output.GLOBAL_ENABLED and
                level <= self.level)

    def fatal(self, message: str) -> None:
        """Log fatal error message"""
        if self._should_output(OutputLevel.FATAL):
            print(self._format_message("FATAL", message), file=sys.stderr)

    def error(self, message: str) -> None:
        """Log error message"""
        if self._should_output(OutputLevel.ERROR):
            print(self._format_message("ERROR", message), file=sys.stderr)

    def warning(self, message: str) -> None:
        """Log warning message"""
        if self._should_output(OutputLevel.WARNING):
            print(self._format_message("WARNING", message))

    def info(self, message: str) -> None:
        """Log info message"""
        if self._should_output(OutputLevel.INFO):
            print(self._format_message("INFO", message))

    def verbose(self, message: str) -> None:
        """Log verbose message"""
        if self._should_output(OutputLevel.VERBOSE):
            print(self._format_message("VERBOSE", message))

    def debug(self, message: str) -> None:
        """Log debug message"""
        if self._should_output(OutputLevel.DEBUG):
            print(self._format_message("DEBUG", message))

    def output(self, message: str, level: OutputLevel = OutputLevel.INFO) -> None:
        """Log message at specified level"""
        if self._should_output(level):
            print(self._format_message(level.name, message))

    @classmethod
    def set_global_level(cls, level: OutputLevel) -> None:
        """Set global default output level"""
        cls.DEFAULT_LEVEL = level

    @classmethod
    def enable_global(cls) -> None:
        """Enable global output"""
        cls.GLOBAL_ENABLED = True

    @classmethod
    def disable_global(cls) -> None:
        """Disable global output"""
        cls.GLOBAL_ENABLED = False
