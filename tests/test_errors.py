import pytest

from core.errors import (
    AuthenticationError,
    BaseInvestWiseError,
    DataNotFoundError,
    PortfolioDataMissingError,
    ProviderQuotaError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
    handle_error,
)
from core.router import Router


class DummySessionManager:
    def __init__(self):
        self._state = {}

    def get_state(self, session_id: str):
        return self._state.get(session_id, {})

    def update_state(self, session_id: str, state_updates: dict):
        current = self._state.get(session_id, {})
        current.update(state_updates)
        self._state[session_id] = current


class RaisingOrchestrator:
    def __init__(self, error: Exception):
        self.error = error

    def think(self, text, settings=None, history=None):
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_status", "expected_reply"),
    [
        (
            ValidationError(),
            "VALIDATION_ERROR",
            400,
            "The request is invalid. Please check your input and try again.",
        ),
        (
            AuthenticationError(),
            "AUTHENTICATION_ERROR",
            401,
            "You need to sign in before continuing.",
        ),
        (
            RateLimitError(),
            "RATE_LIMIT_EXCEEDED",
            429,
            "Too many requests. Please wait a moment and try again.",
        ),
        (
            ProviderQuotaError(),
            "PROVIDER_QUOTA_EXCEEDED",
            429,
            "The AI provider is temporarily unavailable due to quota limits. Please try again shortly.",
        ),
        (
            DataNotFoundError(),
            "DATA_NOT_FOUND",
            404,
            "The requested data could not be found.",
        ),
        (
            PortfolioDataMissingError(),
            "PORTFOLIO_DATA_MISSING",
            400,
            "Portfolio details are required before this operation can continue.",
        ),
        (
            ServiceUnavailableError(),
            "SERVICE_UNAVAILABLE",
            503,
            "The service is temporarily unavailable. Please try again later.",
        ),
        (
            BaseInvestWiseError(),
            "INTERNAL_ERROR",
            500,
            "Something went wrong. Please try again later.",
        ),
    ],
)
def test_handle_error_for_all_investwise_error_types(
    error, expected_code, expected_status, expected_reply
):
    response = handle_error(error)

    assert response["type"] == "error"
    assert response["reply"] == expected_reply
    assert response["data"]["error"]["message"] == error.message
    assert response["data"]["error"]["error_code"] == expected_code
    assert response["data"]["error"]["http_status_code"] == expected_status
    assert (
        response["data"]["error"]["user_friendly_message"]
        == expected_reply
    )


def test_handle_error_preserves_optional_details():
    response = handle_error(
        PortfolioDataMissingError(
            message="Weights are required for portfolio optimization.",
            details={"field": "holdings", "operation": "optimize"},
        )
    )

    assert response["data"]["error"]["message"] == "Weights are required for portfolio optimization."
    assert response["data"]["error"]["details"] == {
        "field": "holdings",
        "operation": "optimize",
    }


def test_handle_error_wraps_generic_exception():
    response = handle_error(RuntimeError("database offline"))

    assert response["type"] == "error"
    assert response["reply"] == "Something went wrong. Please try again later."
    assert response["data"]["error"]["message"] == "database offline"
    assert response["data"]["error"]["error_code"] == "INTERNAL_ERROR"
    assert response["data"]["error"]["http_status_code"] == 500
    assert response["data"]["error"]["details"] == {
        "exception_type": "RuntimeError",
        "original_error": "database offline",
    }


def test_router_uses_standardized_error_payload():
    router = Router(
        orchestrator=RaisingOrchestrator(
            PortfolioDataMissingError(
                details={"field": "holdings"},
            )
        ),
        session_manager=DummySessionManager(),
    )

    response = router.handle_request("session-1", "optimize my portfolio")

    assert response["type"] == "error"
    assert response["reply"] == "Portfolio details are required before this operation can continue."
    assert response["data"]["error"]["error_code"] == "PORTFOLIO_DATA_MISSING"
    assert response["data"]["error"]["http_status_code"] == 400
    assert response["data"]["error"]["details"] == {"field": "holdings"}
