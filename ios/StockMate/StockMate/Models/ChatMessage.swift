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
struct TradingPlanResponse: Codable, Equatable {
    let symbol: String
    let bias: String
    let thesis: String
    let originalThesis: String?  // Preserved from plan creation
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

    // External sentiment (from web/social search)
    let newsSummary: String?  // Brief summary of recent news/catalysts
    let socialSentiment: String?  // bullish, bearish, neutral, mixed, none
    let socialBuzz: String?  // Summary of social discussion
    let sentimentSource: String?  // "reddit" or "x" - which platform was searched

    // V2 Position Management fields
    let positionRecommendation: String?  // "hold", "trim", "reduce", "exit", or null
    let whatToWatch: [String]  // Actionable items
    let riskWarnings: [String]  // Risk warnings
    let alternatives: [AlternativePlan]  // Alternative trade styles

    enum CodingKeys: String, CodingKey {
        case symbol, bias, thesis, status, confidence
        case originalThesis = "original_thesis"
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
        case socialSentiment = "social_sentiment"
        case socialBuzz = "social_buzz"
        case sentimentSource = "sentiment_source"
        case positionRecommendation = "position_recommendation"
        case whatToWatch = "what_to_watch"
        case riskWarnings = "risk_warnings"
        case alternatives
    }

    // Default initializer for optional arrays
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try container.decode(String.self, forKey: .symbol)
        bias = try container.decode(String.self, forKey: .bias)
        thesis = try container.decode(String.self, forKey: .thesis)
        originalThesis = try container.decodeIfPresent(String.self, forKey: .originalThesis)
        entryZoneLow = try container.decodeIfPresent(Double.self, forKey: .entryZoneLow)
        entryZoneHigh = try container.decodeIfPresent(Double.self, forKey: .entryZoneHigh)
        stopLoss = try container.decodeIfPresent(Double.self, forKey: .stopLoss)
        stopReasoning = try container.decodeIfPresent(String.self, forKey: .stopReasoning) ?? ""
        target1 = try container.decodeIfPresent(Double.self, forKey: .target1)
        target2 = try container.decodeIfPresent(Double.self, forKey: .target2)
        target3 = try container.decodeIfPresent(Double.self, forKey: .target3)
        targetReasoning = try container.decodeIfPresent(String.self, forKey: .targetReasoning) ?? ""
        riskReward = try container.decodeIfPresent(Double.self, forKey: .riskReward)
        keySupports = try container.decodeIfPresent([Double].self, forKey: .keySupports) ?? []
        keyResistances = try container.decodeIfPresent([Double].self, forKey: .keyResistances) ?? []
        invalidationCriteria = try container.decodeIfPresent(String.self, forKey: .invalidationCriteria) ?? ""
        technicalSummary = try container.decodeIfPresent(String.self, forKey: .technicalSummary) ?? ""
        status = try container.decodeIfPresent(String.self, forKey: .status) ?? "active"
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ""
        lastEvaluation = try container.decodeIfPresent(String.self, forKey: .lastEvaluation)
        evaluationNotes = try container.decodeIfPresent(String.self, forKey: .evaluationNotes)
        tradeStyle = try container.decodeIfPresent(String.self, forKey: .tradeStyle)
        tradeStyleReasoning = try container.decodeIfPresent(String.self, forKey: .tradeStyleReasoning)
        holdingPeriod = try container.decodeIfPresent(String.self, forKey: .holdingPeriod)
        confidence = try container.decodeIfPresent(Int.self, forKey: .confidence)
        newsSummary = try container.decodeIfPresent(String.self, forKey: .newsSummary)
        socialSentiment = try container.decodeIfPresent(String.self, forKey: .socialSentiment)
        socialBuzz = try container.decodeIfPresent(String.self, forKey: .socialBuzz)
        sentimentSource = try container.decodeIfPresent(String.self, forKey: .sentimentSource)
        positionRecommendation = try container.decodeIfPresent(String.self, forKey: .positionRecommendation)
        whatToWatch = try container.decodeIfPresent([String].self, forKey: .whatToWatch) ?? []
        riskWarnings = try container.decodeIfPresent([String].self, forKey: .riskWarnings) ?? []
        alternatives = try container.decodeIfPresent([AlternativePlan].self, forKey: .alternatives) ?? []
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
        (socialSentiment != nil && socialSentiment!.lowercased() != "none")
    }

    var socialSentimentDisplay: String {
        guard let sentiment = socialSentiment?.lowercased() else { return "" }
        switch sentiment {
        case "bullish": return "Bullish"
        case "bearish": return "Bearish"
        case "neutral": return "Neutral"
        case "mixed": return "Mixed"
        default: return ""
        }
    }

    var socialSentimentColor: String {
        guard let sentiment = socialSentiment?.lowercased() else { return "gray" }
        switch sentiment {
        case "bullish": return "green"
        case "bearish": return "red"
        case "neutral": return "gray"
        case "mixed": return "orange"
        default: return "gray"
        }
    }

    /// Returns the display label for the sentiment source (e.g., "X" or "Reddit")
    var sentimentSourceLabel: String {
        guard let source = sentimentSource?.lowercased() else { return "Social" }
        switch source {
        case "x", "twitter": return "X"
        case "reddit": return "Reddit"
        default: return "Social"
        }
    }

    // V2 Position recommendation helpers
    var hasPositionRecommendation: Bool {
        positionRecommendation != nil && !positionRecommendation!.isEmpty
    }

    var positionRecommendationDisplay: String {
        guard let rec = positionRecommendation?.lowercased() else { return "" }
        switch rec {
        case "hold": return "HOLD"
        case "trim": return "TRIM"
        case "reduce": return "REDUCE"
        case "exit": return "EXIT"
        case "add": return "ADD"
        default: return rec.uppercased()
        }
    }

    var hasWatchItems: Bool {
        !whatToWatch.isEmpty
    }

    var hasRiskWarnings: Bool {
        !riskWarnings.isEmpty
    }

    var hasAlternatives: Bool {
        !alternatives.isEmpty
    }
}

// MARK: - Alternative Plan (V2)

/// Alternative trade style analysis from the v2 sub-agent system
struct AlternativePlan: Codable, Identifiable, Equatable {
    var id: String { tradeStyle }

    let tradeStyle: String  // "day", "swing", "position"
    let bias: String  // "bullish", "bearish", "neutral"
    let suitable: Bool
    let confidence: Int
    let holdingPeriod: String
    let briefThesis: String  // Full thesis text (was truncated, now full)
    let whyNotSelected: String
    let riskReward: Double?
    let positionRecommendation: String?
    let riskWarnings: [String]

    // Additional fields from full SubAgentReport (V2)
    let thesis: String?  // Full thesis (same as briefThesis now)
    let entryZoneLow: Double?
    let entryZoneHigh: Double?
    let entryReasoning: String?
    let stopLoss: Double?
    let stopReasoning: String?
    let target1: Double?
    let target2: Double?
    let target3: Double?
    let targetReasoning: String?
    let whatToWatch: [String]?
    let setupExplanation: String?
    let invalidationCriteria: String?
    let technicalSummary: String?

    enum CodingKeys: String, CodingKey {
        case tradeStyle = "trade_style"
        case bias, suitable, confidence
        case holdingPeriod = "holding_period"
        case briefThesis = "brief_thesis"
        case whyNotSelected = "why_not_selected"
        case riskReward = "risk_reward"
        case positionRecommendation = "position_recommendation"
        case riskWarnings = "risk_warnings"
        // V2 full report fields
        case thesis
        case entryZoneLow = "entry_zone_low"
        case entryZoneHigh = "entry_zone_high"
        case entryReasoning = "entry_reasoning"
        case stopLoss = "stop_loss"
        case stopReasoning = "stop_reasoning"
        case target1 = "target_1"
        case target2 = "target_2"
        case target3 = "target_3"
        case targetReasoning = "target_reasoning"
        case whatToWatch = "what_to_watch"
        case setupExplanation = "setup_explanation"
        case invalidationCriteria = "invalidation_criteria"
        case technicalSummary = "technical_summary"
    }

    // Default initializer for optional arrays
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        tradeStyle = try container.decode(String.self, forKey: .tradeStyle)
        bias = try container.decode(String.self, forKey: .bias)
        suitable = try container.decodeIfPresent(Bool.self, forKey: .suitable) ?? false
        confidence = try container.decodeIfPresent(Int.self, forKey: .confidence) ?? 0
        holdingPeriod = try container.decodeIfPresent(String.self, forKey: .holdingPeriod) ?? ""
        // Try thesis first (full), fall back to brief_thesis
        let fullThesis = try container.decodeIfPresent(String.self, forKey: .thesis)
        let brief = try container.decodeIfPresent(String.self, forKey: .briefThesis)
        briefThesis = fullThesis ?? brief ?? ""
        thesis = fullThesis
        whyNotSelected = try container.decodeIfPresent(String.self, forKey: .whyNotSelected) ?? ""
        riskReward = try container.decodeIfPresent(Double.self, forKey: .riskReward)
        positionRecommendation = try container.decodeIfPresent(String.self, forKey: .positionRecommendation)
        riskWarnings = try container.decodeIfPresent([String].self, forKey: .riskWarnings) ?? []
        // V2 full report fields
        entryZoneLow = try container.decodeIfPresent(Double.self, forKey: .entryZoneLow)
        entryZoneHigh = try container.decodeIfPresent(Double.self, forKey: .entryZoneHigh)
        entryReasoning = try container.decodeIfPresent(String.self, forKey: .entryReasoning)
        stopLoss = try container.decodeIfPresent(Double.self, forKey: .stopLoss)
        stopReasoning = try container.decodeIfPresent(String.self, forKey: .stopReasoning)
        target1 = try container.decodeIfPresent(Double.self, forKey: .target1)
        target2 = try container.decodeIfPresent(Double.self, forKey: .target2)
        target3 = try container.decodeIfPresent(Double.self, forKey: .target3)
        targetReasoning = try container.decodeIfPresent(String.self, forKey: .targetReasoning)
        whatToWatch = try container.decodeIfPresent([String].self, forKey: .whatToWatch)
        setupExplanation = try container.decodeIfPresent(String.self, forKey: .setupExplanation)
        invalidationCriteria = try container.decodeIfPresent(String.self, forKey: .invalidationCriteria)
        technicalSummary = try container.decodeIfPresent(String.self, forKey: .technicalSummary)
    }

    /// Whether this alternative has full report data (V2)
    var hasFullReport: Bool {
        entryZoneLow != nil || entryZoneHigh != nil || stopLoss != nil || target1 != nil
    }

    var tradeStyleDisplay: String {
        switch tradeStyle.lowercased() {
        case "day": return "Day Trade"
        case "swing": return "Swing Trade"
        case "position": return "Position Trade"
        default: return tradeStyle.capitalized
        }
    }

    var tradeStyleIcon: String {
        switch tradeStyle.lowercased() {
        case "day": return "bolt.fill"
        case "swing": return "chart.line.uptrend.xyaxis"
        case "position": return "calendar"
        default: return "chart.bar.fill"
        }
    }

    var biasColor: String {
        switch bias.lowercased() {
        case "bullish": return "green"
        case "bearish": return "red"
        default: return "gray"
        }
    }

    var hasPositionRecommendation: Bool {
        positionRecommendation != nil && !positionRecommendation!.isEmpty
    }

    var positionRecommendationDisplay: String {
        guard let rec = positionRecommendation?.lowercased() else { return "" }
        switch rec {
        case "hold": return "HOLD"
        case "trim": return "TRIM"
        case "reduce": return "REDUCE"
        case "exit": return "EXIT"
        case "add": return "ADD"
        default: return rec.uppercased()
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
