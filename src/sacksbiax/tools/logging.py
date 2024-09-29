__all__ = ["BasicLogger"]
from ..data import LogLevel
from datetime import datetime
import traceback


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class BasicLogger:
    __slots__ = ["level"]
    level: LogLevel

    def __init__(self, level: LogLevel) -> None:
        self.level = level

    def print(self, msg: str, level: LogLevel):
        print(f"{now()}[{level.name:5}]>>> {msg}")

    def debug(self, msg: str):
        if self.level >= LogLevel.DEBUG:
            self.print(msg, LogLevel.DEBUG)

    def info(self, msg: str):
        if self.level >= LogLevel.INFO:
            self.print(msg, LogLevel.INFO)

    def warn(self, msg: str):
        if self.level >= LogLevel.WARN:
            self.print(msg, LogLevel.WARN)

    def error(self, msg: str):
        if self.level >= LogLevel.ERROR:
            self.print(msg, LogLevel.ERROR)

    def fatal(self, msg: str):
        if self.level >= LogLevel.FATAL:
            self.print(msg, LogLevel.FATAL)

    def exception(self, e: Exception):
        print(traceback.format_exc())
        print(e)
