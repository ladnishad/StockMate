"""Portfolio Agent - Orchestrates stock subagents for portfolio-wide analysis.

The Portfolio Agent:
- Has awareness of the user's watchlist
- Dynamically creates stock subagent instances for each watchlist stock
- Can answer portfolio-wide questions ("How is my portfolio doing?")
- Delegates to stock subagents for individual stock analysis
- Persists conversation history to ConversationStore
"""

import logging
import json
from typing import Dict, Any, List, Optional

import anthropic

from app.config import get_settings
from app.storage.watchlist_store import get_watchlist_store
from app.storage.conversation_store import get_conversation_store
from app.storage.position_store import get_position_store
from app.agent.planning_agent import StockPlanningAgent
from app.agent.tools import (
    get_current_price,
    get_market_context,
    get_position_status,
)

logger = logging.getLogger(__name__)


PORTFOLIO_AGENT_SYSTEM = """You are a Portfolio Trading Assistant for StockMate. You help users understand their portfolio and make informed trading decisions.

## Your Capabilities
1. **Portfolio Overview**: Summarize the user's watchlist with current prices, positions, and performance
2. **Stock Analysis**: Delegate to stock-specific subagents for detailed analysis of individual stocks
3. **Market Context**: Provide overall market direction and how it affects the portfolio
4. **Position Management**: Track and advise on open positions across all stocks

## Available Tools
- **get_portfolio_summary**: Get a summary of all watchlist stocks with current prices and positions
- **analyze_stock**: Get comprehensive analysis for a specific stock (delegates to specialized stock agent)
- **get_market_context**: Get overall market direction and index performance
- **get_stock_price**: Get the current price quote for a specific stock

## Communication Style
- Be concise but thorough
- Reference specific prices and levels
- When discussing multiple stocks, use clear formatting
- Remind users this is not financial advice when discussing buy/sell decisions

## Important Guidelines
- Always check the user's watchlist context before answering portfolio questions
- If asked about a specific stock, use the analyze_stock tool to get detailed information
- For portfolio-wide questions, use get_portfolio_summary to aggregate data
- Consider market context when giving advice
"""


class PortfolioAgent:
    """Main portfolio agent that orchestrates stock subagents."""

    def __init__(self, user_id: str = "default"):
        """Initialize the portfolio agent.

        Args:
            user_id: User identifier for accessing watchlist and positions
        """
        self.user_id = user_id
        self._client: Optional[anthropic.Anthropic] = None
        self._watchlist_store = get_watchlist_store()
        self._conversation_store = get_conversation_store()
        self._position_store = get_position_store()

        # Cache for stock subagents
        self._stock_agents: Dict[str, StockPlanningAgent] = {}

    def _get_client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            settings = get_settings()
            if not settings.claude_api_key:
                raise ValueError("CLAUDE_API_KEY not configured")
            self._client = anthropic.Anthropic(api_key=settings.claude_api_key)
        return self._client

    def _get_stock_agent(self, symbol: str) -> StockPlanningAgent:
        """Get or create a stock subagent for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            StockPlanningAgent instance for the symbol
        """
        symbol = symbol.upper()
        if symbol not in self._stock_agents:
            self._stock_agents[symbol] = StockPlanningAgent(symbol, self.user_id)
        return self._stock_agents[symbol]

    def _get_watchlist_symbols(self) -> List[str]:
        """Get all symbols in user's watchlist.

        Returns:
            List of stock symbols
        """
        items = self._watchlist_store.get_watchlist(self.user_id)
        return [item["symbol"] for item in items]

    async def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all watchlist stocks with prices and positions.

        Returns:
            Dictionary with portfolio summary including stocks, positions, and P&L
        """
        symbols = self._get_watchlist_symbols()

        if not symbols:
            return {
                "stocks": [],
                "total_positions": 0,
                "total_unrealized_pnl": 0.0,
                "watchlist_count": 0,
                "message": "Your watchlist is empty. Add stocks to track them.",
            }

        stocks = []
        total_positions = 0
        total_unrealized_pnl = 0.0

        # Limit to 10 stocks for performance
        for symbol in symbols[:10]:
            try:
                price_data = await get_current_price(symbol)
                position_data = await get_position_status(symbol, self.user_id)

                stock_info = {
                    "symbol": symbol,
                    "price": price_data.get("price"),
                    "has_position": position_data.get("has_position", False),
                }

                if position_data.get("has_position"):
                    total_positions += 1
                    stock_info["position_status"] = position_data.get("status")
                    stock_info["entry_price"] = position_data.get("entry_price")
                    stock_info["current_size"] = position_data.get("current_size")
                    stock_info["unrealized_pnl"] = position_data.get("unrealized_pnl")
                    stock_info["unrealized_pnl_pct"] = position_data.get("unrealized_pnl_pct")
                    if position_data.get("unrealized_pnl"):
                        total_unrealized_pnl += position_data["unrealized_pnl"]

                stocks.append(stock_info)

            except Exception as e:
                logger.warning(f"Error getting data for {symbol}: {e}")
                stocks.append({"symbol": symbol, "error": str(e)})

        return {
            "stocks": stocks,
            "total_positions": total_positions,
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "watchlist_count": len(symbols),
        }

    async def _analyze_stock(
        self, symbol: str, question: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a specific stock using its subagent.

        Args:
            symbol: Stock ticker symbol
            question: Optional specific question about the stock

        Returns:
            Dictionary with stock analysis results
        """
        symbol = symbol.upper()
        agent = self._get_stock_agent(symbol)

        try:
            # Gather comprehensive data for the stock
            await agent.gather_comprehensive_data()

            # Get the stock's current summary
            summary = {
                "symbol": symbol,
                "market_data": agent._market_data,
                "technical_data": agent._technical_data,
                "key_levels": agent._key_levels,
            }

            # If there's a specific question, use the agent's chat method
            if question:
                response = await agent.chat(question)
                summary["analysis"] = response

            # Add plan status if exists
            plan = await agent._plan_store.get_plan(self.user_id, symbol)
            if plan:
                summary["has_plan"] = True
                summary["plan_status"] = plan.status
                summary["entry_zone"] = {
                    "low": plan.entry_zone_low,
                    "high": plan.entry_zone_high,
                }
                summary["stop_loss"] = plan.stop_loss
                summary["targets"] = [plan.target_1, plan.target_2, plan.target_3]

            return summary

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for the portfolio agent.

        Returns:
            List of tool definitions for Claude API
        """
        return [
            {
                "name": "get_portfolio_summary",
                "description": "Get a summary of all stocks in the user's watchlist with current prices, positions, and P&L. Use this to answer questions about the overall portfolio.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "analyze_stock",
                "description": "Get detailed analysis for a specific stock including price, technical indicators, key levels, and trading plan. This delegates to a specialized stock agent for comprehensive analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, NVDA, TSLA)",
                        },
                        "question": {
                            "type": "string",
                            "description": "Optional specific question about the stock",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_market_context",
                "description": "Get overall market direction and major index performance (SPY, QQQ, DIA, IWM). Use this to understand the current market environment.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_stock_price",
                "description": "Get the current price quote for a specific stock including bid/ask spread.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        ]

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "get_portfolio_summary":
            return await self._get_portfolio_summary()
        elif tool_name == "analyze_stock":
            return await self._analyze_stock(
                tool_input["symbol"], tool_input.get("question")
            )
        elif tool_name == "get_market_context":
            return await get_market_context()
        elif tool_name == "get_stock_price":
            return await get_current_price(tool_input["symbol"])
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _build_context(self) -> str:
        """Build context string with watchlist info.

        Returns:
            Context string for system prompt
        """
        symbols = self._get_watchlist_symbols()

        if not symbols:
            return "User's watchlist is empty."

        symbol_list = ", ".join(symbols[:10])
        extra = f" (and {len(symbols) - 10} more)" if len(symbols) > 10 else ""

        return f"""## User's Watchlist
Stocks: {symbol_list}{extra}

Use the get_portfolio_summary tool to get current prices and positions for these stocks.
Use the analyze_stock tool to get detailed analysis for any specific stock.
"""

    async def chat(self, user_message: str) -> str:
        """Chat with the portfolio agent.

        Args:
            user_message: User's message/question

        Returns:
            Agent's response
        """
        logger.info(f"Portfolio agent chat: {user_message[:100]}...")

        # Load conversation history
        conversation = await self._conversation_store.get_conversation(
            self.user_id, "portfolio"
        )

        # Build context
        context = await self._build_context()

        system_prompt = f"""{PORTFOLIO_AGENT_SYSTEM}

---

{context}
"""

        # Build messages with history
        messages = conversation.to_claude_messages(max_messages=10)
        messages.append({"role": "user", "content": user_message})

        client = self._get_client()
        settings = get_settings()

        try:
            # Initial request with tools
            response = client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=2000,
                system=system_prompt,
                messages=messages,
                tools=self._build_tools(),
            )

            # Handle tool use loop
            tool_calls = 0
            max_tool_calls = 5  # Prevent infinite loops

            while response.stop_reason == "tool_use" and tool_calls < max_tool_calls:
                tool_calls += 1

                # Extract tool use blocks
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ]

                # Execute tools and build results
                tool_results = []
                for tool_block in tool_use_blocks:
                    logger.info(f"Executing tool: {tool_block.name}")
                    result = await self._execute_tool(tool_block.name, tool_block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": json.dumps(result, default=str),
                        }
                    )

                # Continue conversation with tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = client.messages.create(
                    model=settings.claude_model_fast,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=messages,
                    tools=self._build_tools(),
                )

            # Extract final text response
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            # Save conversation (only user message and final response)
            conversation.add_message("user", user_message)
            conversation.add_message("assistant", response_text)
            await self._conversation_store.save_conversation(conversation)

            return response_text

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Sorry, I encountered an API error: {e}"
        except Exception as e:
            logger.error(f"Portfolio agent error: {e}")
            return f"Sorry, I encountered an error: {e}"

    async def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary without chat.

        Returns:
            Portfolio summary dictionary
        """
        return await self._get_portfolio_summary()


# Singleton cache for portfolio agents (one per user)
_portfolio_agents: Dict[str, PortfolioAgent] = {}


def get_portfolio_agent(user_id: str = "default") -> PortfolioAgent:
    """Get or create portfolio agent instance for a user.

    Args:
        user_id: User identifier

    Returns:
        PortfolioAgent instance
    """
    global _portfolio_agents
    if user_id not in _portfolio_agents:
        _portfolio_agents[user_id] = PortfolioAgent(user_id)
    return _portfolio_agents[user_id]
