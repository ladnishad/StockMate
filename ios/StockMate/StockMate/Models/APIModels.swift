import Foundation

// MARK: - Market Quick Response

struct MarketQuickResponse: Codable {
    let indices: [MarketIndexResponse]
    let marketDirection: String
    let upCount: Int
    let downCount: Int
    let averageChangePct: Double
    let timestamp: String

    enum CodingKeys: String, CodingKey {
        case indices
        case marketDirection = "market_direction"
        case upCount = "up_count"
        case downCount = "down_count"
        case averageChangePct = "average_change_pct"
        case timestamp
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

// MARK: - Profiles Response

struct ProfilesResponse: Codable {
    let profiles: [ProfileInfo]
    let count: Int
    let usage: String
}

struct ProfileInfo: Codable {
    let type: String
    let name: String
    let description: String
    let timeframes: TimeframeInfo
    let holdingPeriod: HoldingPeriodInfo
    let allowedTradeTypes: [String]
    let risk: RiskInfo
    let targets: TargetInfo
    let thresholds: ThresholdInfo
    let weights: [String: Double]

    enum CodingKeys: String, CodingKey {
        case type
        case name
        case description
        case timeframes
        case holdingPeriod = "holding_period"
        case allowedTradeTypes = "allowed_trade_types"
        case risk
        case targets
        case thresholds
        case weights
    }
}

struct TimeframeInfo: Codable {
    let primary: String
    let confirmation: String?
    let entry: String?
}

struct HoldingPeriodInfo: Codable {
    let min: String
    let max: String
}

struct RiskInfo: Codable {
    let riskPercentage: Double
    let stopMethod: String
    let atrMultiplier: Double
    let maxPositionPercent: Double

    enum CodingKeys: String, CodingKey {
        case riskPercentage = "risk_percentage"
        case stopMethod = "stop_method"
        case atrMultiplier = "atr_multiplier"
        case maxPositionPercent = "max_position_percent"
    }
}

struct TargetInfo: Codable {
    let method: String
    let rrRatios: [Double]
    let useFibonacciExtensions: Bool
    let validateAgainstResistance: Bool

    enum CodingKeys: String, CodingKey {
        case method
        case rrRatios = "rr_ratios"
        case useFibonacciExtensions = "use_fibonacci_extensions"
        case validateAgainstResistance = "validate_against_resistance"
    }
}

struct ThresholdInfo: Codable {
    let buyConfidence: Double
    let rsiOverbought: Double
    let rsiOversold: Double
    let adxTrend: Double

    enum CodingKeys: String, CodingKey {
        case buyConfidence = "buy_confidence"
        case rsiOverbought = "rsi_overbought"
        case rsiOversold = "rsi_oversold"
        case adxTrend = "adx_trend"
    }
}

// MARK: - Error Response

struct APIError: Codable, Error {
    let detail: String
}
