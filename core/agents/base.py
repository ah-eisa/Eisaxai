from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from core.persona import Persona, EISAX_PERSONA

class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    """
    
    def __init__(self, name: str, persona: Optional[Persona] = None):
        self.name = name
        self.persona = persona or EISAX_PERSONA

    @abstractmethod
    def think(self, 
              message: str, 
              context: Dict[str, Any], 
              settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message and return a response.
        
        Args:
            message: The user's input message.
            context: Session context (memory, history, etc.).
            settings: Request-specific settings (model, temp, etc.).
            
        Returns:
            Dict containing 'type', 'reply', and optional 'data'.
        """
        pass
