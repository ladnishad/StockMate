import Foundation

/// Comprehensive stock detail for the detail page
struct StockDetail: Codable {
    let symbol: String
    let name: String
    let currentPrice: Double
    let change: Double
    let changePct: Double

    // Key statistics
    let openPrice: Double?
    let highPrice: Double?
    let lowPrice: Double?
    let volume: Int?
    let fiftyTwoWeekHigh: Double?
    let fiftyTwoWeekLow: Double?
    let avgVolume: Int?

    // Analysis (deprecated - use TradingPlanViewModel for AI analysis)
    // These fields now return default values from the backend
    let score: Double  // Deprecated: always 0.0, use plan.confidence instead
    let recommendation: String  // Deprecated: always "PENDING", use plan.bias instead
    let reasoning: String  // Deprecated
    let reasons: [String]  // Deprecated
    let tradePlan: TradePlanDetail?  // Deprecated

    // Multi-timeframe chart data
    let bars1d: [PriceBar]
    let bars1h: [PriceBar]
    let bars15m: [PriceBar]

    // Key levels for chart overlays
    let supportLevels: [Double]
    let resistanceLevels: [Double]

    // Indicators
    let ema9: [Double]
    let ema21: [Double]
    let vwap: Double?

    let timestamp: String

    // MARK: - Computed Properties

    var isUp: Bool {
        change >= 0
    }

    var formattedPrice: String {
        String(format: "$%.2f", currentPrice)
    }

    var formattedChange: String {
        let sign = change >= 0 ? "+" : ""
        return "\(sign)\(String(format: "%.2f", change))"
    }

    var formattedChangePct: String {
        let sign = changePct >= 0 ? "+" : ""
        return "\(sign)\(String(format: "%.2f", changePct))%"
    }

    /// Deprecated: This property always returns false since the backend
    /// no longer performs rule-based analysis. Use TradingPlanViewModel.plan.bias instead.
    var isRecommendedBuy: Bool {
        recommendation.uppercased() == "BUY"
    }

    // Key stats formatted properties
    var formattedOpen: String {
        guard let openPrice = openPrice else { return "—" }
        return String(format: "$%.2f", openPrice)
    }

    var formattedHigh: String {
        guard let highPrice = highPrice else { return "—" }
        return String(format: "$%.2f", highPrice)
    }

    var formattedLow: String {
        guard let lowPrice = lowPrice else { return "—" }
        return String(format: "$%.2f", lowPrice)
    }

    var formattedVolume: String {
        guard let volume = volume else { return "—" }
        return formatLargeNumber(volume)
    }

    var formatted52WeekHigh: String {
        guard let high = fiftyTwoWeekHigh else { return "—" }
        return String(format: "$%.2f", high)
    }

    var formatted52WeekLow: String {
        guard let low = fiftyTwoWeekLow else { return "—" }
        return String(format: "$%.2f", low)
    }

    var formattedAvgVolume: String {
        guard let avgVol = avgVolume else { return "—" }
        return formatLargeNumber(avgVol)
    }

    private func formatLargeNumber(_ number: Int) -> String {
        let absNumber = abs(number)
        if absNumber >= 1_000_000_000 {
            return String(format: "%.2fB", Double(number) / 1_000_000_000)
        } else if absNumber >= 1_000_000 {
            return String(format: "%.2fM", Double(number) / 1_000_000)
        } else if absNumber >= 1_000 {
            return String(format: "%.2fK", Double(number) / 1_000)
        }
        return "\(number)"
    }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case symbol, name, reasoning, reasons, timestamp, vwap, volume
        case currentPrice = "current_price"
        case change
        case changePct = "change_pct"
        // Key statistics
        case openPrice = "open_price"
        case highPrice = "high_price"
        case lowPrice = "low_price"
        case fiftyTwoWeekHigh = "fifty_two_week_high"
        case fiftyTwoWeekLow = "fifty_two_week_low"
        case avgVolume = "avg_volume"
        // Analysis
        case score, recommendation
        case tradePlan = "trade_plan"
        case bars1d = "bars_1d"
        case bars1h = "bars_1h"
        case bars15m = "bars_15m"
        case supportLevels = "support_levels"
        case resistanceLevels = "resistance_levels"
        case ema9 = "ema_9"
        case ema21 = "ema_21"
    }

    // MARK: - Sample Data

    static let sample = StockDetail(
        symbol: "AAPL",
        name: "Apple Inc.",
        currentPrice: 175.50,
        change: 2.35,
        changePct: 1.36,
        openPrice: 173.50,
        highPrice: 176.25,
        lowPrice: 172.80,
        volume: 45_000_000,
        fiftyTwoWeekHigh: 199.62,
        fiftyTwoWeekLow: 164.08,
        avgVolume: 52_000_000,
        // Deprecated analysis fields - now return default values
        score: 0.0,
        recommendation: "PENDING",
        reasoning: "",
        reasons: [],
        tradePlan: nil,
        bars1d: PriceBar.samples,
        bars1h: [],
        bars15m: [],
        supportLevels: [170.00, 165.00],
        resistanceLevels: [180.00, 185.00],
        ema9: [],
        ema21: [],
        vwap: 174.25,
        timestamp: "2025-01-10T10:30:00Z"
    )
}

/// Trade plan detail from stock analysis
struct TradePlanDetail: Codable {
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
    var formattedStopLoss: String { String(format: "$%.2f", stopLoss) }

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

    static let sample = TradePlanDetail(
        tradeType: "swing",
        entryPrice: 175.50,
        stopLoss: 172.00,
        target1: 180.00,
        target2: 185.00,
        target3: 190.00,
        positionSize: 28,
        riskAmount: 100.0,
        riskPercentage: 1.0
    )
}

/// Single price bar for charting
struct PriceBar: Identifiable, Codable {
    var id: String { timestamp }

    let timestamp: String
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Int

    var isUp: Bool {
        close >= open
    }

    var date: Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: timestamp) ?? ISO8601DateFormatter().date(from: timestamp)
    }

    // MARK: - Sample Data

    static let samples: [PriceBar] = {
        let baseDate = Date()
        var bars: [PriceBar] = []
        var price = 170.0

        for i in 0..<30 {
            let dayOffset = TimeInterval(-i * 86400)
            let date = baseDate.addingTimeInterval(dayOffset)
            let formatter = ISO8601DateFormatter()
            let timestamp = formatter.string(from: date)

            let change = Double.random(in: -3...3)
            let open = price
            let close = price + change
            let high = max(open, close) + Double.random(in: 0...2)
            let low = min(open, close) - Double.random(in: 0...2)
            let volume = Int.random(in: 10_000_000...50_000_000)

            bars.append(PriceBar(
                timestamp: timestamp,
                open: open,
                high: high,
                low: low,
                close: close,
                volume: volume
            ))

            price = close
        }

        return bars.reversed()
    }()
}

/// Chart display type selection
enum ChartType: String, CaseIterable, Identifiable {
    case line = "Line"
    case candlestick = "Candle"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .line: return "chart.line.uptrend.xyaxis"
        case .candlestick: return "chart.bar.xaxis"
        }
    }
}

/// Chart timeframe selection
enum ChartTimeframe: String, CaseIterable, Identifiable {
    case oneDay = "1D"
    case oneWeek = "1W"
    case oneMonth = "1M"
    case threeMonths = "3M"
    case sixMonths = "6M"
    case oneYear = "1Y"
    case yearToDate = "YTD"
    case fiveYears = "5Y"
    case all = "ALL"

    var id: String { rawValue }

    var displayName: String {
        rawValue
    }

    /// API value to send to backend /stock/{symbol}/bars endpoint
    var apiValue: String {
        switch self {
        case .oneDay: return "1d"
        case .oneWeek: return "1w"
        case .oneMonth: return "1m"
        case .threeMonths: return "3m"
        case .sixMonths: return "6m"
        case .oneYear: return "1y"
        case .yearToDate: return "ytd"
        case .fiveYears: return "5y"
        case .all: return "all"
        }
    }
}
