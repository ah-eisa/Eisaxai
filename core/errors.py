from __future__ import annotations

from typing import Any, Dict, Optional


class BaseInvestWiseError(Exception):
    default_message = "An unexpected internal error occurred."
    default_error_code = "INTERNAL_ERROR"
    default_http_status_code = 500
    default_user_friendly_message = "Something went wrong. Please try again later."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        error_code: Optional[str] = None,
        http_status_code: Optional[int] = None,
        user_friendly_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message or self.default_message
        self.error_code = error_code or self.default_error_code
        self.http_status_code = http_status_code or self.default_http_status_code
        self.user_friendly_message = (
            user_friendly_message or self.default_user_friendly_message
        )
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "message": self.message,
            "error_code": self.error_code,
            "http_status_code": self.http_status_code,
            "user_friendly_message": self.user_friendly_message,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


class ValidationError(BaseInvestWiseError):
    default_message = "Validation failed."
    default_error_code = "VALIDATION_ERROR"
    default_http_status_code = 400
    default_user_friendly_message = "The request is invalid. Please check your input and try again."


class AuthenticationError(BaseInvestWiseError):
    default_message = "Authentication failed."
    default_error_code = "AUTHENTICATION_ERROR"
    default_http_status_code = 401
    default_user_friendly_message = "You need to sign in before continuing."


class RateLimitError(BaseInvestWiseError):
    default_message = "Rate limit exceeded."
    default_error_code = "RATE_LIMIT_EXCEEDED"
    default_http_status_code = 429
    default_user_friendly_message = "Too many requests. Please wait a moment and try again."


class ProviderQuotaError(RateLimitError):
    default_message = "Provider quota exceeded."
    default_error_code = "PROVIDER_QUOTA_EXCEEDED"
    default_http_status_code = 429
    default_user_friendly_message = "The AI provider is temporarily unavailable due to quota limits. Please try again shortly."


class DataNotFoundError(BaseInvestWiseError):
    default_message = "Requested data was not found."
    default_error_code = "DATA_NOT_FOUND"
    default_http_status_code = 404
    default_user_friendly_message = "The requested data could not be found."


class PortfolioDataMissingError(ValidationError):
    default_message = "Portfolio data is missing."
    default_error_code = "PORTFOLIO_DATA_MISSING"
    default_http_status_code = 400
    default_user_friendly_message = "Portfolio details are required before this operation can continue."


class ServiceUnavailableError(BaseInvestWiseError):
    default_message = "Service is temporarily unavailable."
    default_error_code = "SERVICE_UNAVAILABLE"
    default_http_status_code = 503
    default_user_friendly_message = "The service is temporarily unavailable. Please try again later."


def handle_error(error: Exception) -> Dict[str, Any]:
    if isinstance(error, BaseInvestWiseError):
        investwise_error = error
    else:
        details = {"exception_type": type(error).__name__}
        if str(error):
            details["original_error"] = str(error)
        investwise_error = BaseInvestWiseError(
            message=str(error) or BaseInvestWiseError.default_message,
            details=details,
        )

    return {
        "type": "error",
        "reply": investwise_error.user_friendly_message,
        "data": {"error": investwise_error.to_dict()},
    }


__all__ = [
    "AuthenticationError",
    "BaseInvestWiseError",
    "DataNotFoundError",
    "PortfolioDataMissingError",
    "ProviderQuotaError",
    "RateLimitError",
    "ServiceUnavailableError",
    "ValidationError",
    "handle_error",
]
