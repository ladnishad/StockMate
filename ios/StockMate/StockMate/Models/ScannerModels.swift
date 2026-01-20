import Foundation

/// Trading style for scanner results
enum TradingStyle: String, Codable, CaseIterable {
    case day
    case swing
    case position

    var displayName: String {
        switch self {
        case .day: return "Day"
        case .swing: return "Swing"
        case .position: return "Position"
        }
    }
}

/// Confidence grade letter for scanner results
enum ConfidenceGrade: String, Codable {
    case aPlus = "A+"
    case a = "A"
    case bPlus = "B+"
    case b = "B"
    case c = "C"

    var displayText: String { rawValue }
}

/// Pattern types detected by scanner
enum PatternType: String, Codable {
    // Gap patterns
    case gapUp = "gap_up"
    case gapDown = "gap_down"
    case gapFill = "gap_fill"

    // Breakout patterns
    case resistanceBreakout = "resistance_breakout"
    case vwapReclaim = "vwap_reclaim"
    case rangeBreakout = "range_breakout"
    case channelBreakout = "channel_breakout"
    case baseBreakout = "base_breakout"

    // Reversal patterns
    case oversoldBounce = "oversold_bounce"
    case panicDipBuy = "panic_dip_buy"
    case failedBreakdown = "failed_breakdown"
    case supportBounce = "support_bounce"

    // Continuation patterns
    case bullFlag = "bull_flag"
    case bearFlag = "bear_flag"
    case pullbackToSupport = "pullback_to_support"
    case fibRetracement = "fib_retracement"
    case higherLow = "higher_low"
    case lowerHigh = "lower_high"
    case trendContinuation = "trend_continuation"

    // Momentum patterns
    case newHod = "new_hod"
    case newLod = "new_lod"
    case momentumSurge = "momentum_surge"

    // Long-term patterns
    case goldenCross = "golden_cross"
    case deathCross = "death_cross"
    case weeklyBreakout = "weekly_breakout"
    case majorSupportTest = "major_support_test"
    case trendlineBounce = "trendline_bounce"

    var displayName: String {
        rawValue.split(separator: "_").map { $0.capitalized }.joined(separator: " ")
    }
}

/// A single scanner result for a detected setup
struct ScannerResult: Identifiable, Codable, Equatable {
    var id: String { "\(symbol)-\(style.rawValue)-\(patternType.rawValue)" }

    let symbol: String
    let style: TradingStyle
    let confidenceGrade: ConfidenceGrade
    let confidenceScore: Double
    let currentPrice: Double
    let description: String
    let patternType: PatternType
    let keyLevels: [String: Double]
    let detectedAt: Date
    let isNew: Bool
    let isWatching: Bool
    let warnings: [String]

    // Additional context
    let volumeMultiple: Double?
    let gapPct: Double?
    let fibLevel: Double?
    let rsiValue: Double?
    let vwap: Double?
    let expiresAt: Date?

    // MARK: - Computed Properties

    var formattedPrice: String {
        String(format: "$%.2f", currentPrice)
    }

    var formattedVolumeMultiple: String? {
        guard let vm = volumeMultiple else { return nil }
        return String(format: "%.1fx", vm)
    }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case symbol, style
        case confidenceGrade = "confidence_grade"
        case confidenceScore = "confidence_score"
        case currentPrice = "current_price"
        case description
        case patternType = "pattern_type"
        case keyLevels = "key_levels"
        case detectedAt = "detected_at"
        case isNew = "is_new"
        case isWatching = "is_watching"
        case warnings
        case volumeMultiple = "volume_multiple"
        case gapPct = "gap_pct"
        case fibLevel = "fib_level"
        case rsiValue = "rsi_value"
        case vwap
        case expiresAt = "expires_at"
    }

    // MARK: - Equatable

    static func == (lhs: ScannerResult, rhs: ScannerResult) -> Bool {
        lhs.id == rhs.id
    }

    // MARK: - Sample Data

    static let sample = ScannerResult(
        symbol: "AAPL",
        style: .day,
        confidenceGrade: .aPlus,
        confidenceScore: 87.5,
        currentPrice: 185.50,
        description: "Bull flag breakout: Testing $186 resistance with 2.1x volume",
        patternType: .resistanceBreakout,
        keyLevels: ["support": 183.00, "resistance": 186.00],
        detectedAt: Date(),
        isNew: true,
        isWatching: false,
        warnings: [],
        volumeMultiple: 2.1,
        gapPct: nil,
        fibLevel: nil,
        rsiValue: 55.0,
        vwap: 184.25,
        expiresAt: nil
    )

    static let samples: [ScannerResult] = [
        ScannerResult(
            symbol: "AAPL",
            style: .day,
            confidenceGrade: .aPlus,
            confidenceScore: 87.5,
            currentPrice: 185.50,
            description: "Gap up 3.2%: 2.1x volume, watching $182 support",
            patternType: .gapUp,
            keyLevels: ["support": 182.00, "resistance": 188.00],
            detectedAt: Date(),
            isNew: true,
            isWatching: false,
            warnings: [],
            volumeMultiple: 2.1,
            gapPct: 3.2,
            fibLevel: nil,
            rsiValue: 58.0,
            vwap: 184.25,
            expiresAt: nil
        ),
        ScannerResult(
            symbol: "NVDA",
            style: .day,
            confidenceGrade: .a,
            confidenceScore: 79.0,
            currentPrice: 142.30,
            description: "VWAP reclaim at $141.50: 1.8x volume",
            patternType: .vwapReclaim,
            keyLevels: ["support": 140.00, "resistance": 145.00],
            detectedAt: Date().addingTimeInterval(-300),
            isNew: false,
            isWatching: true,
            warnings: [],
            volumeMultiple: 1.8,
            gapPct: nil,
            fibLevel: nil,
            rsiValue: 52.0,
            vwap: 141.50,
            expiresAt: nil
        ),
        ScannerResult(
            symbol: "MSFT",
            style: .swing,
            confidenceGrade: .bPlus,
            confidenceScore: 72.0,
            currentPrice: 432.15,
            description: "Bull flag forming: 5 days of consolidation near $435",
            patternType: .bullFlag,
            keyLevels: ["support": 428.00, "resistance": 435.00],
            detectedAt: Date().addingTimeInterval(-3600),
            isNew: true,
            isWatching: false,
            warnings: ["Earnings in 5 days"],
            volumeMultiple: 1.2,
            gapPct: nil,
            fibLevel: nil,
            rsiValue: 48.0,
            vwap: nil,
            expiresAt: nil
        ),
        ScannerResult(
            symbol: "AMZN",
            style: .position,
            confidenceGrade: .b,
            confidenceScore: 62.0,
            currentPrice: 198.75,
            description: "Golden cross forming: 50 EMA crossing above 200 EMA",
            patternType: .goldenCross,
            keyLevels: ["support": 195.00, "resistance": 205.00],
            detectedAt: Date().addingTimeInterval(-7200),
            isNew: false,
            isWatching: false,
            warnings: [],
            volumeMultiple: 0.9,
            gapPct: nil,
            fibLevel: nil,
            rsiValue: 55.0,
            vwap: nil,
            expiresAt: nil
        ),
    ]
}

/// Response for scanner endpoint containing results for a style
struct ScannerResponse: Codable {
    let style: TradingStyle
    let results: [ScannerResult]
    let scanTime: Date
    let nextScheduledScan: Date?
    let totalStocksScanned: Int

    enum CodingKeys: String, CodingKey {
        case style, results
        case scanTime = "scan_time"
        case nextScheduledScan = "next_scheduled_scan"
        case totalStocksScanned = "total_stocks_scanned"
    }
}

/// Response containing results from all three scanner styles
struct AllScannersResponse: Codable {
    let day: ScannerResponse
    let swing: ScannerResponse
    let position: ScannerResponse
    let scanTime: Date
    let nextScheduledScan: Date?

    enum CodingKeys: String, CodingKey {
        case day, swing, position
        case scanTime = "scan_time"
        case nextScheduledScan = "next_scheduled_scan"
    }
}

/// Response for scanner status endpoint
struct ScannerStatusResponse: Codable {
    let lastScanTime: Date?
    let nextScheduledScan: Date?
    let currentScanName: String?
    let isScanning: Bool
    let totalResults: [String: Int]

    enum CodingKeys: String, CodingKey {
        case lastScanTime = "last_scan_time"
        case nextScheduledScan = "next_scheduled_scan"
        case currentScanName = "current_scan_name"
        case isScanning = "is_scanning"
        case totalResults = "total_results"
    }
}
