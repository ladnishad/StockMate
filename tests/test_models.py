"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError
from datetime import datetime

from app.models.request import AnalysisRequest
from app.models.response import AnalysisResponse, TradePlan
from app.models.data import PriceBar, Fundamentals, Sentiment, Indicator


class TestAnalysisRequest:
    """Tests for AnalysisRequest model."""

    def test_valid_request(self):
        """Test creating valid request."""
        request = AnalysisRequest(
            symbol="AAPL",
            account_size=10000.0,
            use_ai=False,
        )

        assert request.symbol == "AAPL"
        assert request.account_size == 10000.0
        assert request.use_ai is False

    def test_symbol_normalization(self):
        """Test symbol is normalized to uppercase."""
        request = AnalysisRequest(
            symbol="aapl",
            account_size=10000.0,
        )

        assert request.symbol == "AAPL"

    def test_invalid_account_size(self):
        """Test validation of account size."""
        with pytest.raises(ValidationError):
            AnalysisRequest(
                symbol="AAPL",
                account_size=-100.0,  # Negative
            )

        with pytest.raises(ValidationError):
            AnalysisRequest(
                symbol="AAPL",
                account_size=0.0,  # Zero
            )

    def test_default_use_ai(self):
        """Test default value for use_ai."""
        request = AnalysisRequest(
            symbol="AAPL",
            account_size=10000.0,
        )

        assert request.use_ai is False


class TestTradePlan:
    """Tests for TradePlan model."""

    def test_valid_trade_plan(self):
        """Test creating valid trade plan."""
        plan = TradePlan(
            trade_type="swing",
            entry_price=175.50,
            stop_loss=172.00,
            target_1=180.00,
            target_2=185.00,
            target_3=190.00,
            position_size=28,
            risk_amount=100.00,
            risk_percentage=1.0,
        )

        assert plan.trade_type == "swing"
        assert plan.entry_price == 175.50
        assert plan.position_size == 28

    def test_optional_targets(self):
        """Test that targets 2 and 3 are optional."""
        plan = TradePlan(
            trade_type="day",
            entry_price=175.50,
            stop_loss=172.00,
            target_1=180.00,
            position_size=28,
            risk_amount=100.00,
            risk_percentage=1.0,
        )

        assert plan.target_2 is None
        assert plan.target_3 is None

    def test_invalid_trade_type(self):
        """Test validation of trade type."""
        with pytest.raises(ValidationError):
            TradePlan(
                trade_type="invalid",  # Not in allowed values
                entry_price=175.50,
                stop_loss=172.00,
                target_1=180.00,
                position_size=28,
                risk_amount=100.00,
                risk_percentage=1.0,
            )

    def test_invalid_prices(self):
        """Test validation of prices."""
        with pytest.raises(ValidationError):
            TradePlan(
                trade_type="swing",
                entry_price=0.0,  # Must be > 0
                stop_loss=172.00,
                target_1=180.00,
                position_size=28,
                risk_amount=100.00,
                risk_percentage=1.0,
            )


class TestAnalysisResponse:
    """Tests for AnalysisResponse model."""

    def test_valid_buy_response(self):
        """Test creating valid BUY response."""
        response = AnalysisResponse(
            symbol="AAPL",
            recommendation="BUY",
            confidence=78.5,
            trade_plan=TradePlan(
                trade_type="swing",
                entry_price=175.50,
                stop_loss=172.00,
                target_1=180.00,
                position_size=28,
                risk_amount=100.00,
                risk_percentage=1.0,
            ),
            reasoning="Strong bullish signals",
            timestamp="2025-11-13T10:30:00Z",
        )

        assert response.recommendation == "BUY"
        assert response.trade_plan is not None

    def test_valid_no_buy_response(self):
        """Test creating valid NO_BUY response."""
        response = AnalysisResponse(
            symbol="AAPL",
            recommendation="NO_BUY",
            confidence=45.2,
            trade_plan=None,
            reasoning="Weak signals",
            timestamp="2025-11-13T10:30:00Z",
        )

        assert response.recommendation == "NO_BUY"
        assert response.trade_plan is None

    def test_invalid_recommendation(self):
        """Test validation of recommendation."""
        with pytest.raises(ValidationError):
            AnalysisResponse(
                symbol="AAPL",
                recommendation="MAYBE",  # Invalid
                confidence=50.0,
                reasoning="Test",
                timestamp="2025-11-13T10:30:00Z",
            )

    def test_confidence_bounds(self):
        """Test confidence is bounded 0-100."""
        with pytest.raises(ValidationError):
            AnalysisResponse(
                symbol="AAPL",
                recommendation="BUY",
                confidence=150.0,  # > 100
                reasoning="Test",
                timestamp="2025-11-13T10:30:00Z",
            )


class TestPriceBar:
    """Tests for PriceBar model."""

    def test_valid_price_bar(self):
        """Test creating valid price bar."""
        bar = PriceBar(
            timestamp=datetime(2025, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )

        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.volume == 1000000

    def test_invalid_prices(self):
        """Test validation of prices."""
        with pytest.raises(ValidationError):
            PriceBar(
                timestamp=datetime(2025, 1, 1),
                open=0.0,  # Must be > 0
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )


class TestSentiment:
    """Tests for Sentiment model."""

    def test_valid_sentiment(self):
        """Test creating valid sentiment."""
        sentiment = Sentiment(
            score=0.5,
            label="bullish",
            news_count=10,
        )

        assert sentiment.score == 0.5
        assert sentiment.label == "bullish"

    def test_invalid_score(self):
        """Test validation of sentiment score."""
        with pytest.raises(ValidationError):
            Sentiment(
                score=2.0,  # Must be -1 to 1
                label="bullish",
                news_count=10,
            )

    def test_invalid_label(self):
        """Test validation of sentiment label."""
        with pytest.raises(ValidationError):
            Sentiment(
                score=0.5,
                label="invalid",  # Must be bearish/neutral/bullish
                news_count=10,
            )
