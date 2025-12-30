import Foundation

/// Represents a single chat message in the conversation
struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    let content: String
    let isUser: Bool
    let timestamp: Date
    var isTyping: Bool = false

    // Optional stock context for AI responses
    var stockContext: StockContext?

    init(
        id: UUID = UUID(),
        content: String,
        isUser: Bool,
        timestamp: Date = Date(),
        isTyping: Bool = false,
        stockContext: StockContext? = nil
    ) {
        self.id = id
        self.content = content
        self.isUser = isUser
        self.timestamp = timestamp
        self.isTyping = isTyping
        self.stockContext = stockContext
    }

    /// Creates a typing indicator placeholder message
    static func typingIndicator() -> ChatMessage {
        ChatMessage(content: "", isUser: false, isTyping: true)
    }
}

/// Stock context shown inline with AI responses
struct StockContext: Equatable {
    let symbol: String
    let price: Double?
    let changePercent: Double?
    let rsi: Double?
    let hasPosition: Bool
    let hasPlan: Bool
    let planStatus: String?

    init(
        symbol: String,
        price: Double? = nil,
        changePercent: Double? = nil,
        rsi: Double? = nil,
        hasPosition: Bool = false,
        hasPlan: Bool = false,
        planStatus: String? = nil
    ) {
        self.symbol = symbol
        self.price = price
        self.changePercent = changePercent
        self.rsi = rsi
        self.hasPosition = hasPosition
        self.hasPlan = hasPlan
        self.planStatus = planStatus
    }

    var priceFormatted: String {
        guard let price = price else { return "—" }
        return String(format: "$%.2f", price)
    }

    var changeFormatted: String {
        guard let change = changePercent else { return "" }
        let sign = change >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, change)
    }

    var isPositive: Bool {
        (changePercent ?? 0) >= 0
    }

    var planStatusFormatted: String {
        guard let status = planStatus else { return "" }
        return status.capitalized
    }
}

/// API response model for chat
struct ChatResponse: Codable {
    let symbol: String?
    let response: String
    let context: ChatResponseContext?
    let marketContext: MarketContextResponse?
    let hasPlan: Bool?
    let planStatus: String?

    enum CodingKeys: String, CodingKey {
        case symbol, response, context
        case marketContext = "market_context"
        case hasPlan = "has_plan"
        case planStatus = "plan_status"
    }
}

struct ChatResponseContext: Codable {
    let hasPosition: Bool?
    let positionStatus: String?
    let marketDirection: String?
    let currentPrice: Double?

    // Evaluation context (present when user triggered an evaluation)
    let justEvaluated: Bool?
    let evaluationStatus: String?
    let priceAtCreation: Double?

    enum CodingKeys: String, CodingKey {
        case hasPosition = "has_position"
        case positionStatus = "position_status"
        case marketDirection = "market_direction"
        case currentPrice = "current_price"
        case justEvaluated = "just_evaluated"
        case evaluationStatus = "evaluation_status"
        case priceAtCreation = "price_at_creation"
    }
}

struct MarketContextResponse: Codable {
    let marketDirection: String?
    let bullishIndices: Int?
    let totalIndices: Int?

    enum CodingKeys: String, CodingKey {
        case marketDirection = "market_direction"
        case bullishIndices = "bullish_indices"
        case totalIndices = "total_indices"
    }
}

// MARK: - Portfolio Chat Response

/// Response from portfolio chat endpoint
struct PortfolioChatResponse: Codable {
    let response: String
    let portfolioSummary: PortfolioSummary?
    let conversationKey: String

    enum CodingKeys: String, CodingKey {
        case response
        case portfolioSummary = "portfolio_summary"
        case conversationKey = "conversation_key"
    }
}

/// Summary of user's portfolio from the Portfolio Agent
struct PortfolioSummary: Codable {
    let stocks: [PortfolioStock]
    let totalPositions: Int
    let totalUnrealizedPnl: Double?
    let watchlistCount: Int
    let message: String?

    enum CodingKeys: String, CodingKey {
        case stocks
        case totalPositions = "total_positions"
        case totalUnrealizedPnl = "total_unrealized_pnl"
        case watchlistCount = "watchlist_count"
        case message
    }
}

/// Individual stock in portfolio summary
struct PortfolioStock: Codable {
    let symbol: String
    let price: Double?
    let hasPosition: Bool?
    let positionStatus: String?
    let entryPrice: Double?
    let currentSize: Int?
    let unrealizedPnl: Double?
    let unrealizedPnlPct: Double?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case symbol, price, error
        case hasPosition = "has_position"
        case positionStatus = "position_status"
        case entryPrice = "entry_price"
        case currentSize = "current_size"
        case unrealizedPnl = "unrealized_pnl"
        case unrealizedPnlPct = "unrealized_pnl_pct"
    }
}

// MARK: - Chat History Response

/// Response from chat history endpoint
struct ChatHistoryResponse: Codable {
    let key: String
    let messages: [ChatHistoryMessage]
    let count: Int
}

/// A single message from server chat history
struct ChatHistoryMessage: Codable {
    let role: String
    let content: String
    let timestamp: String?
}

/// Response from plan evaluation
struct EvaluationResponse: Codable {
    let symbol: String
    let evaluation: String
    let planStatus: String
    let currentPrice: Double?
    let priceAtCreation: Double?
    let adjustmentsMade: [String]
    let previousValues: [String: Double]
    let newValues: [String: Double]

    enum CodingKeys: String, CodingKey {
        case symbol, evaluation
        case planStatus = "plan_status"
        case currentPrice = "current_price"
        case priceAtCreation = "price_at_creation"
        case adjustmentsMade = "adjustments_made"
        case previousValues = "previous_values"
        case newValues = "new_values"
    }

    /// Check if a specific field was adjusted
    func wasAdjusted(_ field: String) -> Bool {
        adjustmentsMade.contains(field)
    }

    /// Get the previous value for a field if it was adjusted
    func previousValue(for field: String) -> Double? {
        previousValues[field]
    }

    /// Get the new value for a field if it was adjusted
    func newValue(for field: String) -> Double? {
        newValues[field]
    }
}

/// Trading plan from the planning agent
struct TradingPlanResponse: Codable {
    let symbol: String
    let bias: String
    let thesis: String
    let entryZoneLow: Double?
    let entryZoneHigh: Double?
    let stopLoss: Double?
    let stopReasoning: String
    let target1: Double?
    let target2: Double?
    let target3: Double?
    let targetReasoning: String
    let riskReward: Double?
    let keySupports: [Double]
    let keyResistances: [Double]
    let invalidationCriteria: String
    let technicalSummary: String
    let status: String
    let createdAt: String
    let lastEvaluation: String?
    let evaluationNotes: String?

    // Trade style fields (agent-determined)
    let tradeStyle: String?  // "day", "swing", "position"
    let tradeStyleReasoning: String?
    let holdingPeriod: String?  // e.g., "1-3 days", "1-2 weeks"
    let confidence: Int?  // 0-100

    // External sentiment (from web search)
    let newsSummary: String?  // Brief summary of recent news/catalysts
    let redditSentiment: String?  // bullish, bearish, neutral, mixed, none
    let redditBuzz: String?  // Summary of Reddit discussion

    enum CodingKeys: String, CodingKey {
        case symbol, bias, thesis, status, confidence
        case entryZoneLow = "entry_zone_low"
        case entryZoneHigh = "entry_zone_high"
        case stopLoss = "stop_loss"
        case stopReasoning = "stop_reasoning"
        case target1 = "target_1"
        case target2 = "target_2"
        case target3 = "target_3"
        case targetReasoning = "target_reasoning"
        case riskReward = "risk_reward"
        case keySupports = "key_supports"
        case keyResistances = "key_resistances"
        case invalidationCriteria = "invalidation_criteria"
        case technicalSummary = "technical_summary"
        case createdAt = "created_at"
        case lastEvaluation = "last_evaluation"
        case evaluationNotes = "evaluation_notes"
        case tradeStyle = "trade_style"
        case tradeStyleReasoning = "trade_style_reasoning"
        case holdingPeriod = "holding_period"
        case newsSummary = "news_summary"
        case redditSentiment = "reddit_sentiment"
        case redditBuzz = "reddit_buzz"
    }

    var isBullish: Bool { bias.lowercased() == "bullish" }
    var isBearish: Bool { bias.lowercased() == "bearish" }
    var isActive: Bool { status.lowercased() == "active" }

    // Trade style helpers
    var tradeStyleDisplay: String {
        guard let style = tradeStyle else { return "Swing Trade" }
        switch style.lowercased() {
        case "day": return "Day Trade"
        case "swing": return "Swing Trade"
        case "position": return "Position Trade"
        default: return "Swing Trade"
        }
    }

    var tradeStyleIcon: String {
        guard let style = tradeStyle else { return "clock" }
        switch style.lowercased() {
        case "day": return "bolt.fill"
        case "swing": return "clock"
        case "position": return "calendar"
        default: return "clock"
        }
    }

    // Sentiment helpers
    var hasNewsSentiment: Bool {
        (newsSummary != nil && !newsSummary!.isEmpty) ||
        (redditSentiment != nil && redditSentiment!.lowercased() != "none")
    }

    var redditSentimentDisplay: String {
        guard let sentiment = redditSentiment?.lowercased() else { return "" }
        switch sentiment {
        case "bullish": return "Bullish"
        case "bearish": return "Bearish"
        case "neutral": return "Neutral"
        case "mixed": return "Mixed"
        default: return ""
        }
    }

    var redditSentimentColor: String {
        guard let sentiment = redditSentiment?.lowercased() else { return "gray" }
        switch sentiment {
        case "bullish": return "green"
        case "bearish": return "red"
        case "neutral": return "gray"
        case "mixed": return "orange"
        default: return "gray"
        }
    }
}

// MARK: - Interactive Plan Session Models

/// Message in a planning conversation
struct PlanMessage: Codable, Identifiable {
    let id: String
    let role: String  // "user", "assistant", "system"
    let content: String
    let messageType: String  // "question", "answer", "adjustment_request", "adjustment_response", "approval", "info"
    let timestamp: String
    let options: [PlanOption]
    let selectedOption: String?

    enum CodingKeys: String, CodingKey {
        case id, role, content, timestamp, options
        case messageType = "message_type"
        case selectedOption = "selected_option"
    }

    var isUser: Bool { role == "user" }
    var isAssistant: Bool { role == "assistant" }
}

/// Option presented by AI for adjustments
struct PlanOption: Codable {
    let label: String
    let description: String
    let value: String?
}

/// Response for a plan session
struct PlanSessionResponse: Codable {
    let sessionId: String
    let status: String  // "generating", "draft", "refining", "approved", "rejected"
    let symbol: String
    let draftPlan: TradingPlanResponse?
    let messages: [PlanMessage]
    let revisionCount: Int
    let createdAt: String
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case symbol, status, messages
        case sessionId = "session_id"
        case draftPlan = "draft_plan"
        case revisionCount = "revision_count"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }

    var isGenerating: Bool { status == "generating" }
    var isDraft: Bool { status == "draft" }
    var isRefining: Bool { status == "refining" }
    var isApproved: Bool { status == "approved" }
    var hasDraftPlan: Bool { draftPlan != nil }
}

/// Response after submitting feedback
struct PlanFeedbackResponse: Codable {
    let aiResponse: String
    let updatedPlan: TradingPlanResponse?
    let options: [PlanOption]
    let sessionStatus: String

    enum CodingKeys: String, CodingKey {
        case options
        case aiResponse = "ai_response"
        case updatedPlan = "updated_plan"
        case sessionStatus = "session_status"
    }
}
