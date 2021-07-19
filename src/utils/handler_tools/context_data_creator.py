"""This module contains an abstract class structure for creating a context data file.
The ContextDataCreator class should organize the creation of the context data file."""
from dataclasses import dataclass, field
import logging
from pathlib import Path
import abc

logger = logging.getLogger(__name__)


@dataclass
class ContextDataCreator(abc.ABC):
    """abstract class that manages the creation of the context data file"""
    dest_file_path: Path

    @abc.abstractmethod
    def manage_features(self) -> None:
        """abstract method to call the feature calculation process."""
        pass
