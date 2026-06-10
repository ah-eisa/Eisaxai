import json
import logging
import sys
import uuid
import datetime
from datetime import timezone
from typing import Any, Dict, Optional

# Configure root logger to output JSON
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "props"):
            log_obj.update(record.props) 
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])

def get_logger(name: str):
    return logging.getLogger(name)

class StructuredLogger:
    def __init__(self, name: str, request_id: Optional[str] = None):
        self.logger = get_logger(name)
        self.request_id = request_id or str(uuid.uuid4())

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra={"request_id": self.request_id, "props": kwargs})

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={"request_id": self.request_id, "props": kwargs})
        
    def warn(self, msg: str, **kwargs):
        self.logger.warning(msg, extra={"request_id": self.request_id, "props": kwargs})
        
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra={"request_id": self.request_id, "props": kwargs})
