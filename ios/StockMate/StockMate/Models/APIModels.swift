import Foundation

// MARK: - Market Status

struct MarketStatus: Codable {
    let isOpen: Bool
    let nextEvent: String?
    let nextEventType: String  // "open" or "close"

    enum CodingKeys: String, CodingKey {
        case isOpen = "is_open"
        case nextEvent = "next_event"
        case nextEventType = "next_event_type"
    }
}

// MARK: - Market Quick Response

struct MarketQuickResponse: Codable {
    let indices: [MarketIndexResponse]
    let marketDirection: String
    let upCount: Int
    let downCount: Int
    let averageChangePct: Double
    let timestamp: String
    let marketStatus: MarketStatus?

    enum CodingKeys: String, CodingKey {
        case indices
        case marketDirection = "market_direction"
        case upCount = "up_count"
        case downCount = "down_count"
        case averageChangePct = "average_change_pct"
        case timestamp
        case marketStatus = "market_status"
    }

    var marketIndices: [MarketIndex] {
        indices.map { $0.toMarketIndex() }
    }
}

struct MarketIndexResponse: Codable {
    let symbol: String
    let name: String
    let price: Double
    let change: Double
    let changePct: Double
    let dayHigh: Double?
    let dayLow: Double?
    let dayVolume: Int?

    enum CodingKeys: String, CodingKey {
        case symbol
        case name
        case price
        case change
        case changePct = "change_pct"
        case dayHigh = "day_high"
        case dayLow = "day_low"
        case dayVolume = "day_volume"
    }

    func toMarketIndex() -> MarketIndex {
        MarketIndex(
            symbol: symbol,
            name: name,
            price: price,
            change: change,
            changePct: changePct,
            dayHigh: dayHigh,
            dayLow: dayLow,
            volume: dayVolume
        )
    }
}

// MARK: - Market Scan Response (for watchlist)

struct MarketScanResponse: Codable {
    let market: MarketOverview
    let sectors: SectorOverview
    let topStocks: [SectorStocks]
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case market
        case sectors
        case topStocks = "top_stocks"
        case timestamp
    }

    /// Flattens all stocks from all sectors into a single sorted list
    var allStocks: [Stock] {
        topStocks.flatMap { sector in
            sector.leaders.map { leader in
                Stock(
                    symbol: leader.symbol,
                    score: leader.score,
                    recommendation: leader.recommendation,
                    currentPrice: leader.currentPrice,
                    reasons: leader.reasons,
                    sectorName: sector.sectorName,
                    tradePlan: leader.tradePlan?.toTradePlan()
                )
            }
        }
        .sorted { $0.score > $1.score }
    }
}

struct MarketOverview: Codable {
    let marketSignal: String
    let bullishCount: Int
    let summary: String

    enum CodingKeys: String, CodingKey {
        case marketSignal = "market_signal"
        case bullishCount = "bullish_count"
        case summary
    }
}

struct SectorOverview: Codable {
    let leadingSectors: [String]
    let rotationSignal: String

    enum CodingKeys: String, CodingKey {
        case leadingSectors = "leading_sectors"
        case rotationSignal = "rotation_signal"
    }
}

struct SectorStocks: Codable {
    let sectorName: String
    let sectorEtf: String
    let leaders: [StockLeader]
    let stocksAnalyzed: Int
    let stocksAboveThreshold: Int
    let averageScore: Double
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case sectorName = "sector_name"
        case sectorEtf = "sector_etf"
        case leaders
        case stocksAnalyzed = "stocks_analyzed"
        case stocksAboveThreshold = "stocks_above_threshold"
        case averageScore = "average_score"
        case timestamp
    }
}

struct StockLeader: Codable {
    let symbol: String
    let score: Double
    let recommendation: String
    let currentPrice: Double
    let reasons: [String]
    let tradePlan: TradePlanResponse?

    enum CodingKeys: String, CodingKey {
        case symbol
        case score
        case recommendation
        case currentPrice = "current_price"
        case reasons
        case tradePlan = "trade_plan"
    }
}

struct TradePlanResponse: Codable {
    let entry: Double?
    let stop: Double?
    let target: Double?

    func toTradePlan() -> TradePlan {
        TradePlan(
            tradeType: "swing",
            entryPrice: entry ?? 0,
            stopLoss: stop ?? 0,
            target1: target ?? 0,
            target2: nil,
            target3: nil,
            positionSize: nil,
            riskAmount: nil,
            riskPercentage: nil
        )
    }
}

// MARK: - Analysis Response

struct AnalysisResponse: Codable {
    let symbol: String
    let recommendation: String
    let confidence: Double
    let tradePlan: TradePlan?
    let reasoning: String
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case symbol
        case recommendation
        case confidence
        case tradePlan = "trade_plan"
        case reasoning
        case timestamp
    }
}

// MARK: - Error Response

struct APIError: Codable, Error {
    let detail: String
}


// MARK: - Smart Analysis Response (Profile-Less AI Analysis)

struct SmartAnalysisResponse: Codable {
    let symbol: String
    let currentPrice: Double
    let recommendation: String
    let tradePlan: EnhancedTradePlan
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case symbol
        case currentPrice = "current_price"
        case recommendation
        case tradePlan = "trade_plan"
        case timestamp
    }

    var isBuy: Bool {
        recommendation == "BUY"
    }
}

struct EnhancedTradePlan: Codable {
    let tradeStyle: TradeStyleRecommendation
    let bias: String
    let thesis: String
    let confidence: Int
    let entryZoneLow: Double?
    let entryZoneHigh: Double?
    let stopLoss: Double?
    let stopReasoning: String
    let targets: [PriceTarget]
    let riskReward: Double?
    let positionSizePct: Double?
    let keySupports: [Double]
    let keyResistances: [Double]
    let invalidationCriteria: String
    let educational: EducationalContent

    enum CodingKeys: String, CodingKey {
        case tradeStyle = "trade_style"
        case bias
        case thesis
        case confidence
        case entryZoneLow = "entry_zone_low"
        case entryZoneHigh = "entry_zone_high"
        case stopLoss = "stop_loss"
        case stopReasoning = "stop_reasoning"
        case targets
        case riskReward = "risk_reward"
        case positionSizePct = "position_size_pct"
        case keySupports = "key_supports"
        case keyResistances = "key_resistances"
        case invalidationCriteria = "invalidation_criteria"
        case educational
    }

    var isBullish: Bool {
        bias == "bullish"
    }

    var isBearish: Bool {
        bias == "bearish"
    }

    var hasEntry: Bool {
        entryZoneLow != nil && entryZoneHigh != nil
    }

    /// Formatted entry zone string
    var formattedEntryZone: String {
        guard let low = entryZoneLow, let high = entryZoneHigh else {
            return "N/A"
        }
        return "$\(String(format: "%.2f", low)) - $\(String(format: "%.2f", high))"
    }

    /// First target price and reasoning
    var primaryTarget: PriceTarget? {
        targets.first
    }
}

struct TradeStyleRecommendation: Codable {
    let recommendedStyle: String
    let reasoning: String
    let holdingPeriod: String

    enum CodingKeys: String, CodingKey {
        case recommendedStyle = "recommended_style"
        case reasoning
        case holdingPeriod = "holding_period"
    }

    var displayName: String {
        switch recommendedStyle {
        case "day": return "Day Trade"
        case "swing": return "Swing Trade"
        case "position": return "Position Trade"
        default: return recommendedStyle.capitalized
        }
    }

    var icon: String {
        switch recommendedStyle {
        case "day": return "bolt.fill"
        case "swing": return "waveform.path.ecg"
        case "position": return "chart.line.uptrend.xyaxis"
        default: return "chart.bar"
        }
    }

    var accentColor: String {
        switch recommendedStyle {
        case "day": return "orange"
        case "swing": return "blue"
        case "position": return "purple"
        default: return "gray"
        }
    }
}

struct PriceTarget: Codable {
    let price: Double
    let reasoning: String

    var formattedPrice: String {
        "$\(String(format: "%.2f", price))"
    }
}

struct EducationalContent: Codable {
    let setupExplanation: String
    let levelExplanations: [String: String]
    let whatToWatch: [String]
    let scenarios: [ScenarioPath]
    let riskWarnings: [String]
    let chartAnnotations: [ChartAnnotation]

    enum CodingKeys: String, CodingKey {
        case setupExplanation = "setup_explanation"
        case levelExplanations = "level_explanations"
        case whatToWatch = "what_to_watch"
        case scenarios
        case riskWarnings = "risk_warnings"
        case chartAnnotations = "chart_annotations"
    }

    var hasContent: Bool {
        !setupExplanation.isEmpty || !scenarios.isEmpty || !whatToWatch.isEmpty
    }

    /// Get the bullish scenario
    var bullishScenario: ScenarioPath? {
        scenarios.first { $0.scenario == "bullish" }
    }

    /// Get the bearish scenario
    var bearishScenario: ScenarioPath? {
        scenarios.first { $0.scenario == "bearish" }
    }

    /// Get the sideways scenario
    var sidewaysScenario: ScenarioPath? {
        scenarios.first { $0.scenario == "sideways" }
    }
}

struct ScenarioPath: Codable, Identifiable {
    let scenario: String
    let probability: Int
    let description: String
    let priceTarget: Double?
    let keyTrigger: String

    enum CodingKeys: String, CodingKey {
        case scenario
        case probability
        case description
        case priceTarget = "price_target"
        case keyTrigger = "key_trigger"
    }

    var id: String { scenario }

    var displayName: String {
        scenario.capitalized
    }

    var icon: String {
        switch scenario {
        case "bullish": return "arrow.up.right"
        case "bearish": return "arrow.down.right"
        case "sideways": return "arrow.left.and.right"
        default: return "questionmark"
        }
    }

    var color: String {
        switch scenario {
        case "bullish": return "green"
        case "bearish": return "red"
        case "sideways": return "gray"
        default: return "gray"
        }
    }
}

struct ChartAnnotation: Codable, Identifiable {
    let type: String
    let price: Double?
    let priceHigh: Double?
    let priceLow: Double?
    let label: String
    let color: String
    let description: String

    enum CodingKeys: String, CodingKey {
        case type
        case price
        case priceHigh = "price_high"
        case priceLow = "price_low"
        case label
        case color
        case description
    }

    var id: String { "\(type)-\(label)-\(price ?? 0)" }
}
