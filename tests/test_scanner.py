"""Tests for Market Scanner feature."""

import pytest
from datetime import datetime, timedelta
from typing import List

from pydantic import ValidationError

from app.models.scanner import (
    ScannerResult,
    ScannerResponse,
    AllScannersResponse,
    ScannerStatusResponse,
    AddFromScannerRequest,
    TradingStyle,
    PatternType,
    ConfidenceGrade,
    ConfidenceScoreComponents,
    score_to_grade,
    VOLUME_THRESHOLDS,
    VOLUME_CONFIDENCE_MULTIPLIER,
    MIN_PRICE,
    EXPIRATION_HOURS,
)
from app.models.data import PriceBar


# =============================================================================
# Model Tests
# =============================================================================


class TestTradingStyle:
    """Tests for TradingStyle enum."""

    def test_trading_styles(self):
        """Test all trading styles exist."""
        assert TradingStyle.DAY.value == "day"
        assert TradingStyle.SWING.value == "swing"
        assert TradingStyle.POSITION.value == "position"

    def test_trading_style_from_string(self):
        """Test creating TradingStyle from string."""
        assert TradingStyle("day") == TradingStyle.DAY
        assert TradingStyle("swing") == TradingStyle.SWING
        assert TradingStyle("position") == TradingStyle.POSITION

    def test_invalid_trading_style(self):
        """Test invalid trading style raises error."""
        with pytest.raises(ValueError):
            TradingStyle("invalid")


class TestPatternType:
    """Tests for PatternType enum."""

    def test_gap_patterns(self):
        """Test gap pattern types exist."""
        assert PatternType.GAP_UP.value == "gap_up"
        assert PatternType.GAP_DOWN.value == "gap_down"
        assert PatternType.GAP_FILL.value == "gap_fill"

    def test_breakout_patterns(self):
        """Test breakout pattern types exist."""
        assert PatternType.RESISTANCE_BREAKOUT.value == "resistance_breakout"
        assert PatternType.VWAP_RECLAIM.value == "vwap_reclaim"
        assert PatternType.RANGE_BREAKOUT.value == "range_breakout"

    def test_reversal_patterns(self):
        """Test reversal pattern types exist."""
        assert PatternType.OVERSOLD_BOUNCE.value == "oversold_bounce"
        assert PatternType.FAILED_BREAKDOWN.value == "failed_breakdown"

    def test_continuation_patterns(self):
        """Test continuation pattern types exist."""
        assert PatternType.BULL_FLAG.value == "bull_flag"
        assert PatternType.FIB_RETRACEMENT.value == "fib_retracement"


class TestConfidenceGrade:
    """Tests for ConfidenceGrade enum and score_to_grade function."""

    def test_confidence_grades(self):
        """Test all confidence grades exist."""
        assert ConfidenceGrade.A_PLUS.value == "A+"
        assert ConfidenceGrade.A.value == "A"
        assert ConfidenceGrade.B_PLUS.value == "B+"
        assert ConfidenceGrade.B.value == "B"
        assert ConfidenceGrade.C.value == "C"

    def test_score_to_grade_a_plus(self):
        """Test A+ grade (85-100)."""
        assert score_to_grade(100) == ConfidenceGrade.A_PLUS
        assert score_to_grade(90) == ConfidenceGrade.A_PLUS
        assert score_to_grade(85) == ConfidenceGrade.A_PLUS

    def test_score_to_grade_a(self):
        """Test A grade (75-84)."""
        assert score_to_grade(84) == ConfidenceGrade.A
        assert score_to_grade(80) == ConfidenceGrade.A
        assert score_to_grade(75) == ConfidenceGrade.A

    def test_score_to_grade_b_plus(self):
        """Test B+ grade (65-74)."""
        assert score_to_grade(74) == ConfidenceGrade.B_PLUS
        assert score_to_grade(70) == ConfidenceGrade.B_PLUS
        assert score_to_grade(65) == ConfidenceGrade.B_PLUS

    def test_score_to_grade_b(self):
        """Test B grade (55-64)."""
        assert score_to_grade(64) == ConfidenceGrade.B
        assert score_to_grade(60) == ConfidenceGrade.B
        assert score_to_grade(55) == ConfidenceGrade.B

    def test_score_to_grade_c(self):
        """Test C grade (45-54 and below)."""
        assert score_to_grade(54) == ConfidenceGrade.C
        assert score_to_grade(50) == ConfidenceGrade.C
        assert score_to_grade(45) == ConfidenceGrade.C
        assert score_to_grade(30) == ConfidenceGrade.C  # Below threshold still returns C


class TestConfidenceScoreComponents:
    """Tests for ConfidenceScoreComponents model."""

    def test_calculate_total_all_max(self):
        """Test calculating total score with all max values."""
        components = ConfidenceScoreComponents(
            pattern_clarity=100,
            volume_confirmation=100,
            indicator_alignment=100,
            risk_reward_ratio=100,
            fib_level_proximity=100,
        )

        total = components.calculate_total()
        assert total == 100  # 25 + 25 + 20 + 15 + 15

    def test_calculate_total_mixed(self):
        """Test calculating total score with mixed values."""
        components = ConfidenceScoreComponents(
            pattern_clarity=80,    # 80 * 0.25 = 20
            volume_confirmation=60,  # 60 * 0.25 = 15
            indicator_alignment=70,  # 70 * 0.20 = 14
            risk_reward_ratio=50,   # 50 * 0.15 = 7.5
            fib_level_proximity=40,  # 40 * 0.15 = 6
        )

        total = components.calculate_total()
        assert total == 62.5  # 20 + 15 + 14 + 7.5 + 6

    def test_calculate_total_all_zero(self):
        """Test calculating total score with all zero values."""
        components = ConfidenceScoreComponents()
        total = components.calculate_total()
        assert total == 0


class TestScannerResult:
    """Tests for ScannerResult model."""

    def test_valid_scanner_result(self):
        """Test creating valid scanner result."""
        result = ScannerResult(
            symbol="AAPL",
            style=TradingStyle.DAY,
            confidence_grade=ConfidenceGrade.A_PLUS,
            confidence_score=87.5,
            current_price=185.50,
            description="Bull flag breakout: Testing $186 resistance with 2.1x volume",
            pattern_type=PatternType.RESISTANCE_BREAKOUT,
            key_levels={"support": 183.00, "resistance": 186.00},
        )

        assert result.symbol == "AAPL"
        assert result.style == TradingStyle.DAY
        assert result.confidence_grade == ConfidenceGrade.A_PLUS
        assert result.confidence_score == 87.5

    def test_scanner_result_defaults(self):
        """Test scanner result default values."""
        result = ScannerResult(
            symbol="TSLA",
            style=TradingStyle.SWING,
            confidence_grade=ConfidenceGrade.B,
            confidence_score=60.0,
            current_price=250.00,
            description="Test pattern",
            pattern_type=PatternType.BULL_FLAG,
        )

        assert result.is_new is True
        assert result.is_watching is False
        assert result.warnings == []
        assert result.key_levels == {}

    def test_scanner_result_optional_fields(self):
        """Test scanner result optional fields."""
        result = ScannerResult(
            symbol="NVDA",
            style=TradingStyle.POSITION,
            confidence_grade=ConfidenceGrade.A,
            confidence_score=78.0,
            current_price=500.00,
            description="Golden cross forming",
            pattern_type=PatternType.GOLDEN_CROSS,
            volume_multiple=1.8,
            gap_pct=3.5,
            fib_level=61.8,
            rsi_value=55.0,
            vwap=495.50,
        )

        assert result.volume_multiple == 1.8
        assert result.gap_pct == 3.5
        assert result.fib_level == 61.8
        assert result.rsi_value == 55.0
        assert result.vwap == 495.50

    def test_scanner_result_validation(self):
        """Test scanner result validation."""
        # Invalid confidence score (above 100)
        with pytest.raises(ValidationError):
            ScannerResult(
                symbol="AAPL",
                style=TradingStyle.DAY,
                confidence_grade=ConfidenceGrade.A_PLUS,
                confidence_score=150.0,  # Invalid
                current_price=185.50,
                description="Test",
                pattern_type=PatternType.GAP_UP,
            )

        # Invalid current price (zero)
        with pytest.raises(ValidationError):
            ScannerResult(
                symbol="AAPL",
                style=TradingStyle.DAY,
                confidence_grade=ConfidenceGrade.A_PLUS,
                confidence_score=85.0,
                current_price=0,  # Invalid
                description="Test",
                pattern_type=PatternType.GAP_UP,
            )


class TestScannerResponse:
    """Tests for ScannerResponse model."""

    def test_valid_scanner_response(self):
        """Test creating valid scanner response."""
        result = ScannerResult(
            symbol="AAPL",
            style=TradingStyle.DAY,
            confidence_grade=ConfidenceGrade.A,
            confidence_score=80.0,
            current_price=185.50,
            description="Test",
            pattern_type=PatternType.GAP_UP,
        )

        response = ScannerResponse(
            style=TradingStyle.DAY,
            results=[result],
            total_stocks_scanned=1500,
        )

        assert response.style == TradingStyle.DAY
        assert len(response.results) == 1
        assert response.total_stocks_scanned == 1500

    def test_scanner_response_empty_results(self):
        """Test scanner response with no results."""
        response = ScannerResponse(
            style=TradingStyle.SWING,
            results=[],
            total_stocks_scanned=1000,
        )

        assert response.results == []
        assert response.total_stocks_scanned == 1000


class TestAllScannersResponse:
    """Tests for AllScannersResponse model."""

    def test_valid_all_scanners_response(self):
        """Test creating valid all scanners response."""
        day_response = ScannerResponse(style=TradingStyle.DAY, results=[])
        swing_response = ScannerResponse(style=TradingStyle.SWING, results=[])
        position_response = ScannerResponse(style=TradingStyle.POSITION, results=[])

        response = AllScannersResponse(
            day=day_response,
            swing=swing_response,
            position=position_response,
        )

        assert response.day.style == TradingStyle.DAY
        assert response.swing.style == TradingStyle.SWING
        assert response.position.style == TradingStyle.POSITION


class TestScannerStatusResponse:
    """Tests for ScannerStatusResponse model."""

    def test_valid_scanner_status(self):
        """Test creating valid scanner status."""
        status = ScannerStatusResponse(
            last_scan_time=datetime.utcnow(),
            next_scheduled_scan=datetime.utcnow() + timedelta(hours=1),
            current_scan_name="market_open",
            is_scanning=False,
            total_results={"day": 10, "swing": 8, "position": 5},
        )

        assert status.current_scan_name == "market_open"
        assert status.is_scanning is False
        assert status.total_results["day"] == 10

    def test_scanner_status_defaults(self):
        """Test scanner status default values."""
        status = ScannerStatusResponse()

        assert status.last_scan_time is None
        assert status.next_scheduled_scan is None
        assert status.is_scanning is False
        assert status.total_results == {"day": 0, "swing": 0, "position": 0}


class TestAddFromScannerRequest:
    """Tests for AddFromScannerRequest model."""

    def test_valid_add_request(self):
        """Test creating valid add from scanner request."""
        request = AddFromScannerRequest(
            scanner_source="Day Trade Scanner",
            scanner_reason="Breakout Setup",
        )

        assert request.scanner_source == "Day Trade Scanner"
        assert request.scanner_reason == "Breakout Setup"


# =============================================================================
# Constants Tests
# =============================================================================


class TestScannerConstants:
    """Tests for scanner constants."""

    def test_volume_thresholds(self):
        """Test volume thresholds are defined correctly."""
        assert VOLUME_THRESHOLDS[TradingStyle.DAY] == 500_000
        assert VOLUME_THRESHOLDS[TradingStyle.SWING] == 200_000
        assert VOLUME_THRESHOLDS[TradingStyle.POSITION] == 100_000

    def test_volume_confidence_multipliers(self):
        """Test volume confidence multipliers are defined correctly."""
        assert VOLUME_CONFIDENCE_MULTIPLIER[TradingStyle.DAY] == 2.0
        assert VOLUME_CONFIDENCE_MULTIPLIER[TradingStyle.SWING] == 1.5
        assert VOLUME_CONFIDENCE_MULTIPLIER[TradingStyle.POSITION] == 1.0

    def test_min_price(self):
        """Test minimum price filter is defined."""
        assert MIN_PRICE == 1.00

    def test_expiration_hours(self):
        """Test expiration hours are defined correctly."""
        assert EXPIRATION_HOURS[TradingStyle.DAY] == 8
        assert EXPIRATION_HOURS[TradingStyle.SWING] == 72
        assert EXPIRATION_HOURS[TradingStyle.POSITION] == 168


# =============================================================================
# Scanner Store Tests
# =============================================================================


class TestScannerStore:
    """Tests for ScannerStore class."""

    @pytest.fixture
    def scanner_store(self, tmp_path):
        """Create a scanner store for testing."""
        from app.storage.scanner_store import ScannerStore

        file_path = tmp_path / "test_scanner.json"
        return ScannerStore(file_path=file_path)

    @pytest.fixture
    def sample_results(self):
        """Create sample scanner results for testing."""
        return [
            ScannerResult(
                symbol="AAPL",
                style=TradingStyle.DAY,
                confidence_grade=ConfidenceGrade.A_PLUS,
                confidence_score=87.5,
                current_price=185.50,
                description="Gap up 3.5%: 2.1x volume",
                pattern_type=PatternType.GAP_UP,
            ),
            ScannerResult(
                symbol="MSFT",
                style=TradingStyle.DAY,
                confidence_grade=ConfidenceGrade.A,
                confidence_score=78.0,
                current_price=380.00,
                description="VWAP reclaim",
                pattern_type=PatternType.VWAP_RECLAIM,
            ),
        ]

    def test_store_and_get_results(self, scanner_store, sample_results):
        """Test storing and retrieving results."""
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
            scan_name="test_scan",
            total_scanned=1000,
        )

        results = scanner_store.get_results(TradingStyle.DAY)
        assert len(results) == 2
        assert results[0].symbol == "AAPL"

    def test_new_detection(self, scanner_store, sample_results):
        """Test new result detection."""
        # First scan
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results[:1],  # Only AAPL
            scan_name="scan_1",
        )

        # Second scan with new stock
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,  # AAPL and MSFT
            scan_name="scan_2",
        )

        results = scanner_store.get_results(TradingStyle.DAY)

        # AAPL should not be new, MSFT should be new
        aapl_result = next(r for r in results if r.symbol == "AAPL")
        msft_result = next(r for r in results if r.symbol == "MSFT")

        assert aapl_result.is_new is False
        assert msft_result.is_new is True

    def test_user_watchlist_marking(self, scanner_store, sample_results):
        """Test marking results as already watching."""
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
        )

        user_watchlist = {"AAPL", "GOOGL"}
        results = scanner_store.get_results(TradingStyle.DAY, user_watchlist)

        aapl_result = next(r for r in results if r.symbol == "AAPL")
        msft_result = next(r for r in results if r.symbol == "MSFT")

        assert aapl_result.is_watching is True
        assert msft_result.is_watching is False

    def test_invalidate_symbol(self, scanner_store, sample_results):
        """Test invalidating a symbol."""
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
        )

        scanner_store.invalidate_symbol("AAPL")

        results = scanner_store.get_results(TradingStyle.DAY)
        assert len(results) == 1
        assert results[0].symbol == "MSFT"

    def test_clear_all(self, scanner_store, sample_results):
        """Test clearing all results."""
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
        )

        scanner_store.clear_all()

        results = scanner_store.get_results(TradingStyle.DAY)
        assert len(results) == 0

    def test_get_status(self, scanner_store, sample_results):
        """Test getting scanner status."""
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
            scan_name="test_scan",
        )

        status = scanner_store.get_status()

        assert status["current_scan_name"] == "test_scan"
        assert status["total_results"]["day"] == 2

    def test_result_by_symbol(self, scanner_store, sample_results):
        """Test finding result by symbol."""
        scanner_store.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
        )

        result = scanner_store.get_result_by_symbol("AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

        result = scanner_store.get_result_by_symbol("NONEXISTENT")
        assert result is None

    def test_persistence(self, tmp_path, sample_results):
        """Test store persists and loads data correctly."""
        from app.storage.scanner_store import ScannerStore

        file_path = tmp_path / "test_scanner.json"

        # Create store, add data, let it save
        store1 = ScannerStore(file_path=file_path)
        store1.store_results(
            style=TradingStyle.DAY,
            results=sample_results,
            scan_name="persistence_test",
        )

        # Create new store instance (should load from file)
        store2 = ScannerStore(file_path=file_path)
        results = store2.get_results(TradingStyle.DAY)

        assert len(results) == 2
        assert store2.get_status()["current_scan_name"] == "persistence_test"


# =============================================================================
# Template Description Tests
# =============================================================================


class TestDescriptionTemplates:
    """Tests for description template generation."""

    def test_day_trade_gap_up_template(self):
        """Test day trade gap up description format."""
        # Gap up template format: "Gap up {gap_pct}%: {volume_multiple}x volume, watching ${support} support"
        from app.services.market_scanner_service import MarketScannerService

        service = MarketScannerService()
        description = service._generate_description(
            style=TradingStyle.DAY,
            pattern_type=PatternType.GAP_UP,
            stock_data={"current_price": 100.0},
            pattern_data={
                "gap_pct": 3.5,
                "volume_multiple": 2.1,
                "support": 97.00,
            },
        )

        assert "Gap up" in description
        assert "3.5%" in description
        assert "2.1x" in description
        assert "$97.00" in description

    def test_swing_trade_bull_flag_template(self):
        """Test swing trade bull flag description format."""
        from app.services.market_scanner_service import MarketScannerService

        service = MarketScannerService()
        description = service._generate_description(
            style=TradingStyle.SWING,
            pattern_type=PatternType.BULL_FLAG,
            stock_data={"current_price": 100.0},
            pattern_data={
                "days": 5,
                "resistance": 105.00,
                "volume_multiple": 1.5,
            },
        )

        assert "Bull flag" in description
        assert "5 days" in description
        assert "$105.00" in description

    def test_position_trade_golden_cross_template(self):
        """Test position trade golden cross description format."""
        from app.services.market_scanner_service import MarketScannerService

        service = MarketScannerService()
        description = service._generate_description(
            style=TradingStyle.POSITION,
            pattern_type=PatternType.GOLDEN_CROSS,
            stock_data={"current_price": 100.0},
            pattern_data={
                "ema_50": 98.00,
                "ema_200": 95.00,
            },
        )

        assert "Golden cross" in description or "50 EMA" in description


# =============================================================================
# Integration Tests (require API access - mark as skip by default)
# =============================================================================


@pytest.mark.skip(reason="Requires Alpaca API access")
class TestScannerServiceIntegration:
    """Integration tests for MarketScannerService."""

    @pytest.fixture
    def scanner_service(self):
        """Create scanner service for testing."""
        from app.services.market_scanner_service import MarketScannerService
        return MarketScannerService()

    @pytest.mark.asyncio
    async def test_full_scan(self, scanner_service):
        """Test running a full scan."""
        result = await scanner_service.run_scan(scan_name="test")

        assert result.day is not None
        assert result.swing is not None
        assert result.position is not None

    @pytest.mark.asyncio
    async def test_get_status(self, scanner_service):
        """Test getting scanner status."""
        status = scanner_service.get_status()

        assert "last_scan_time" in status.__dict__
        assert "is_scanning" in status.__dict__


# =============================================================================
# Watchlist Metadata Tests
# =============================================================================


class TestWatchlistScannerMetadata:
    """Tests for scanner metadata in watchlist."""

    def test_watchlist_item_scanner_fields(self):
        """Test WatchlistItem includes scanner metadata fields."""
        from app.models.watchlist import WatchlistItem

        item = WatchlistItem(
            symbol="AAPL",
            scanner_source="Day Trade Scanner",
            scanner_reason="Breakout Setup",
        )

        assert item.scanner_source == "Day Trade Scanner"
        assert item.scanner_reason == "Breakout Setup"

    def test_watchlist_store_scanner_metadata(self, tmp_path):
        """Test WatchlistStore handles scanner metadata."""
        from app.storage.watchlist_store import WatchlistStore

        file_path = tmp_path / "test_watchlist.json"
        store = WatchlistStore(file_path=file_path)

        # Add symbol with scanner metadata
        result = store.add_symbol(
            user_id="test_user",
            symbol="AAPL",
            scanner_source="Day Trade Scanner",
            scanner_reason="Gap Play",
        )

        assert result["scanner_source"] == "Day Trade Scanner"
        assert result["scanner_reason"] == "Gap Play"

        # Retrieve and verify
        items = store.get_watchlist("test_user")
        assert len(items) == 1
        assert items[0]["scanner_source"] == "Day Trade Scanner"
        assert items[0]["scanner_reason"] == "Gap Play"
