"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.models.response import AnalysisResponse, TradePlan


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_buy_response():
    """Mock BUY recommendation response."""
    return AnalysisResponse(
        symbol="AAPL",
        recommendation="BUY",
        confidence=78.5,
        trade_plan=TradePlan(
            trade_type="swing",
            entry_price=175.50,
            stop_loss=172.00,
            target_1=180.00,
            target_2=185.00,
            target_3=190.00,
            position_size=28,
            risk_amount=100.00,
            risk_percentage=1.0,
        ),
        reasoning="Bullish sentiment | Price above key EMAs",
        timestamp="2025-11-13T10:30:00Z",
    )


@pytest.fixture
def mock_no_buy_response():
    """Mock NO_BUY recommendation response."""
    return AnalysisResponse(
        symbol="AAPL",
        recommendation="NO_BUY",
        confidence=45.2,
        trade_plan=None,
        reasoning="Bearish sentiment | Confidence too low",
        timestamp="2025-11-13T10:30:00Z",
    )


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "StockMate API"
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "alpaca_configured" in data


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    @patch("app.main.run_analysis")
    def test_analyze_success_buy(self, mock_run_analysis, client, mock_buy_response):
        """Test successful analysis with BUY recommendation."""
        mock_run_analysis.return_value = mock_buy_response

        response = client.post(
            "/analyze",
            json={
                "symbol": "AAPL",
                "account_size": 10000.0,
                "use_ai": False,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["symbol"] == "AAPL"
        assert data["recommendation"] == "BUY"
        assert data["confidence"] == 78.5
        assert data["trade_plan"] is not None
        assert data["trade_plan"]["trade_type"] == "swing"
        assert data["trade_plan"]["entry_price"] == 175.50
        assert data["trade_plan"]["position_size"] == 28

    @patch("app.main.run_analysis")
    def test_analyze_success_no_buy(self, mock_run_analysis, client, mock_no_buy_response):
        """Test successful analysis with NO_BUY recommendation."""
        mock_run_analysis.return_value = mock_no_buy_response

        response = client.post(
            "/analyze",
            json={
                "symbol": "AAPL",
                "account_size": 10000.0,
                "use_ai": False,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["symbol"] == "AAPL"
        assert data["recommendation"] == "NO_BUY"
        assert data["confidence"] == 45.2
        assert data["trade_plan"] is None

    def test_analyze_invalid_symbol(self, client):
        """Test analysis with invalid symbol."""
        response = client.post(
            "/analyze",
            json={
                "symbol": "",  # Empty symbol
                "account_size": 10000.0,
                "use_ai": False,
            }
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_invalid_account_size(self, client):
        """Test analysis with invalid account size."""
        response = client.post(
            "/analyze",
            json={
                "symbol": "AAPL",
                "account_size": -100.0,  # Negative account size
                "use_ai": False,
            }
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_missing_required_fields(self, client):
        """Test analysis with missing required fields."""
        response = client.post(
            "/analyze",
            json={
                "symbol": "AAPL",
                # Missing account_size
            }
        )

        assert response.status_code == 422  # Validation error

    @patch("app.main.run_analysis")
    def test_analyze_api_error(self, mock_run_analysis, client):
        """Test analysis when API error occurs."""
        mock_run_analysis.side_effect = Exception("API connection failed")

        response = client.post(
            "/analyze",
            json={
                "symbol": "AAPL",
                "account_size": 10000.0,
                "use_ai": False,
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_analyze_symbol_normalization(self, client):
        """Test that symbol is normalized (uppercase)."""
        with patch("app.main.run_analysis") as mock_run_analysis:
            mock_run_analysis.return_value = MagicMock(
                symbol="AAPL",
                recommendation="NO_BUY",
                confidence=50.0,
                trade_plan=None,
                reasoning="Test",
                timestamp="2025-11-13T10:30:00Z",
            )

            response = client.post(
                "/analyze",
                json={
                    "symbol": "aapl",  # lowercase
                    "account_size": 10000.0,
                    "use_ai": False,
                }
            )

            # Symbol should be normalized to uppercase
            mock_run_analysis.assert_called_once()
            call_args = mock_run_analysis.call_args
            assert call_args[1]["symbol"] == "AAPL"


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
