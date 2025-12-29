"""Background service for periodic plan evaluation.

Evaluates trading plans every 30 minutes during market hours,
and triggers immediate evaluation when price approaches key levels.
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, Set, Optional
from dataclasses import dataclass

from app.services.scheduler import is_market_open
from app.storage.plan_store import get_plan_store, TradingPlan
from app.storage.watchlist_store import get_watchlist_store
from app.agent.planning_agent import StockPlanningAgent
from app.agent.tools import get_current_price

logger = logging.getLogger(__name__)

# Evaluation intervals
PERIODIC_EVAL_MINUTES = 30  # Evaluate all plans every 30 minutes
KEY_LEVEL_THRESHOLD_PCT = 0.02  # 2% from key level triggers evaluation


@dataclass
class EvaluationResult:
    """Result of a plan evaluation."""
    symbol: str
    status: str  # valid, adjust, invalidated
    evaluation: str
    current_price: float
    triggered_by: str  # periodic, key_level, manual, market_close


class PlanEvaluator:
    """Background service that evaluates trading plans periodically."""

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._plan_store = get_plan_store()
        self._watchlist_store = get_watchlist_store()

        # Track last evaluation times
        self._last_evaluation: Dict[str, datetime] = {}

        # Track last known prices for key level detection
        self._last_prices: Dict[str, float] = {}

        # Symbols being actively monitored
        self._monitored_symbols: Set[str] = set()

        # Evaluation results
        self._recent_results: Dict[str, EvaluationResult] = {}

        # Track market state for close detection
        self._was_market_open: bool = False

    async def start(self, user_id: str = "default"):
        """Start the background evaluation service."""
        if self._running:
            logger.warning("PlanEvaluator already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._evaluation_loop(user_id))
        logger.info("PlanEvaluator started")

    async def stop(self):
        """Stop the background evaluation service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PlanEvaluator stopped")

    async def _evaluation_loop(self, user_id: str):
        """Main evaluation loop."""
        while self._running:
            try:
                market_open = is_market_open()

                # Detect market close transition: was open, now closed
                if self._was_market_open and not market_open:
                    logger.info("Market just closed - running final evaluation for all plans")
                    active_plans = await self._plan_store.get_active_plans(user_id)
                    for plan in active_plans:
                        await self._evaluate_plan(plan, user_id, "market_close")
                    self._was_market_open = False

                # If market is closed, wait
                if not market_open:
                    await asyncio.sleep(60)  # Check every minute if market opened
                    continue

                # Market is open - update state
                self._was_market_open = True

                # Get all active plans
                active_plans = await self._plan_store.get_active_plans(user_id)

                for plan in active_plans:
                    self._monitored_symbols.add(plan.symbol)

                    # Check if periodic evaluation is due
                    if self._should_evaluate_periodic(plan.symbol):
                        await self._evaluate_plan(plan, user_id, "periodic")

                    # Check for key level proximity
                    elif await self._is_near_key_level(plan):
                        await self._evaluate_plan(plan, user_id, "key_level")

                # Sleep before next check (1 minute between price checks)
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(60)

    def _should_evaluate_periodic(self, symbol: str) -> bool:
        """Check if periodic evaluation is due for a symbol."""
        last_eval = self._last_evaluation.get(symbol)
        if not last_eval:
            return True

        elapsed = (datetime.utcnow() - last_eval).total_seconds() / 60
        return elapsed >= PERIODIC_EVAL_MINUTES

    async def _is_near_key_level(self, plan: TradingPlan) -> bool:
        """Check if current price is near a key level."""
        try:
            price_data = await get_current_price(plan.symbol)
            current_price = price_data.get("price")

            if not current_price:
                return False

            # Store last price
            self._last_prices[plan.symbol] = current_price

            # Check proximity to key levels
            key_levels = []

            # Entry zone
            if plan.entry_zone_low:
                key_levels.append(plan.entry_zone_low)
            if plan.entry_zone_high:
                key_levels.append(plan.entry_zone_high)

            # Stop loss
            if plan.stop_loss:
                key_levels.append(plan.stop_loss)

            # Targets
            if plan.target_1:
                key_levels.append(plan.target_1)
            if plan.target_2:
                key_levels.append(plan.target_2)
            if plan.target_3:
                key_levels.append(plan.target_3)

            # Key supports/resistances
            key_levels.extend(plan.key_supports[:2])  # Top 2 supports
            key_levels.extend(plan.key_resistances[:2])  # Top 2 resistances

            # Check if price is within threshold of any level
            for level in key_levels:
                if level and level > 0:
                    distance_pct = abs(current_price - level) / current_price
                    if distance_pct <= KEY_LEVEL_THRESHOLD_PCT:
                        logger.info(
                            f"{plan.symbol} within {distance_pct*100:.1f}% of ${level:.2f}"
                        )
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking key levels for {plan.symbol}: {e}")
            return False

    async def _evaluate_plan(
        self,
        plan: TradingPlan,
        user_id: str,
        triggered_by: str
    ) -> Optional[EvaluationResult]:
        """Evaluate a single plan."""
        try:
            logger.info(f"Evaluating plan for {plan.symbol} (triggered by: {triggered_by})")

            agent = StockPlanningAgent(plan.symbol, user_id)
            result = await agent.evaluate_plan()

            # Update last evaluation time
            self._last_evaluation[plan.symbol] = datetime.utcnow()

            # Create result
            eval_result = EvaluationResult(
                symbol=plan.symbol,
                status=result.get("plan_status", "unknown"),
                evaluation=result.get("evaluation", ""),
                current_price=result.get("current_price", 0),
                triggered_by=triggered_by
            )

            self._recent_results[plan.symbol] = eval_result

            logger.info(f"Evaluated {plan.symbol}: {eval_result.status}")
            return eval_result

        except Exception as e:
            logger.error(f"Error evaluating plan for {plan.symbol}: {e}")
            return None

    async def evaluate_now(self, symbol: str, user_id: str = "default") -> Optional[EvaluationResult]:
        """Manually trigger evaluation for a symbol."""
        plan = await self._plan_store.get_plan(user_id, symbol.upper())
        if not plan:
            return None
        return await self._evaluate_plan(plan, user_id, "manual")

    def get_recent_results(self) -> Dict[str, EvaluationResult]:
        """Get recent evaluation results."""
        return self._recent_results.copy()

    def get_status(self) -> Dict:
        """Get evaluator status."""
        return {
            "running": self._running,
            "monitored_symbols": list(self._monitored_symbols),
            "last_evaluations": {
                sym: dt.isoformat()
                for sym, dt in self._last_evaluation.items()
            },
            "recent_results": {
                sym: {
                    "status": r.status,
                    "triggered_by": r.triggered_by,
                    "current_price": r.current_price
                }
                for sym, r in self._recent_results.items()
            }
        }


# Singleton instance
_evaluator: Optional[PlanEvaluator] = None


def get_plan_evaluator() -> PlanEvaluator:
    """Get singleton PlanEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = PlanEvaluator()
    return _evaluator
