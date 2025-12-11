import Foundation

/// Represents a stock in the watchlist with analysis data
struct Stock: Identifiable, Codable, Equatable {
    var id: String { symbol }
    let symbol: String
    let score: Double
    let recommendation: String
    let currentPrice: Double
    let reasons: [String]
    let sectorName: String?
    let tradePlan: TradePlan?

    var isRecommendedBuy: Bool {
        recommendation.uppercased() == "BUY"
    }

    var formattedPrice: String {
        String(format: "$%.2f", currentPrice)
    }

    var formattedScore: String {
        String(format: "%.0f", score)
    }

    var scoreColor: ScoreColor {
        if score >= 75 { return .excellent }
        if score >= 65 { return .good }
        if score >= 50 { return .neutral }
        return .weak
    }

    enum ScoreColor {
        case excellent, good, neutral, weak

        var description: String {
            switch self {
            case .excellent: return "Excellent"
            case .good: return "Good"
            case .neutral: return "Neutral"
            case .weak: return "Weak"
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case symbol
        case score
        case recommendation
        case currentPrice = "current_price"
        case reasons
        case sectorName = "sector_name"
        case tradePlan = "trade_plan"
    }

    // Sample data for previews
    static let sample = Stock(
        symbol: "NVDA",
        score: 82.5,
        recommendation: "BUY",
        currentPrice: 142.65,
        reasons: ["Strong momentum", "Volume confirmation", "Above key EMAs"],
        sectorName: "Technology",
        tradePlan: TradePlan.sample
    )

    static let samples: [Stock] = [
        Stock(symbol: "NVDA", score: 82.5, recommendation: "BUY", currentPrice: 142.65, reasons: ["Strong momentum", "Volume confirmation"], sectorName: "Technology", tradePlan: TradePlan.sample),
        Stock(symbol: "AAPL", score: 76.3, recommendation: "BUY", currentPrice: 248.92, reasons: ["MACD crossover", "Fibonacci support"], sectorName: "Technology", tradePlan: nil),
        Stock(symbol: "MSFT", score: 71.8, recommendation: "BUY", currentPrice: 432.15, reasons: ["Bullish trend", "RSI healthy"], sectorName: "Technology", tradePlan: nil),
        Stock(symbol: "META", score: 68.2, recommendation: "BUY", currentPrice: 612.30, reasons: ["Sector leadership"], sectorName: "Communication", tradePlan: nil),
        Stock(symbol: "AMZN", score: 64.5, recommendation: "NO_BUY", currentPrice: 227.85, reasons: ["Resistance ahead"], sectorName: "Consumer", tradePlan: nil)
    ]
}

/// Represents a trade plan from the analysis
struct TradePlan: Codable, Equatable {
    let tradeType: String
    let entryPrice: Double
    let stopLoss: Double
    let target1: Double
    let target2: Double?
    let target3: Double?
    let positionSize: Int?
    let riskAmount: Double?
    let riskPercentage: Double?

    var formattedEntry: String { String(format: "$%.2f", entryPrice) }
    var formattedStop: String { String(format: "$%.2f", stopLoss) }
    var formattedStopLoss: String { String(format: "$%.2f", stopLoss) }
    var formattedTarget1: String { String(format: "$%.2f", target1) }

    /// All targets as an array (filtering out nil values)
    var targets: [Double] {
        [target1, target2, target3].compactMap { $0 }
    }

    var riskRewardRatio: Double? {
        let risk = entryPrice - stopLoss
        guard risk > 0 else { return nil }
        let reward = target1 - entryPrice
        return reward / risk
    }

    enum CodingKeys: String, CodingKey {
        case tradeType = "trade_type"
        case entryPrice = "entry_price"
        case stopLoss = "stop_loss"
        case target1 = "target_1"
        case target2 = "target_2"
        case target3 = "target_3"
        case positionSize = "position_size"
        case riskAmount = "risk_amount"
        case riskPercentage = "risk_percentage"
    }

    static let sample = TradePlan(
        tradeType: "swing",
        entryPrice: 141.50,
        stopLoss: 138.00,
        target1: 148.00,
        target2: 152.00,
        target3: 158.00,
        positionSize: 28,
        riskAmount: 100.0,
        riskPercentage: 1.0
    )
}
