import SwiftUI

/// Available trader profiles for customized analysis
enum TraderProfile: String, CaseIterable, Identifiable, Codable {
    case dayTrader = "day_trader"
    case swingTrader = "swing_trader"
    case positionTrader = "position_trader"
    case longTermInvestor = "long_term_investor"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .dayTrader: return "Day"
        case .swingTrader: return "Swing"
        case .positionTrader: return "Position"
        case .longTermInvestor: return "Long-Term"
        }
    }

    var fullName: String {
        switch self {
        case .dayTrader: return "Day Trader"
        case .swingTrader: return "Swing Trader"
        case .positionTrader: return "Position Trader"
        case .longTermInvestor: return "Long-Term Investor"
        }
    }

    var icon: String {
        switch self {
        case .dayTrader: return "bolt.fill"
        case .swingTrader: return "waveform.path.ecg"
        case .positionTrader: return "chart.line.uptrend.xyaxis"
        case .longTermInvestor: return "building.columns.fill"
        }
    }

    var description: String {
        switch self {
        case .dayTrader:
            return "Intraday trades with tight stops. Focus on VWAP and volume."
        case .swingTrader:
            return "Multi-day holds using Fibonacci levels and structure-based entries."
        case .positionTrader:
            return "Multi-week positions following strong trends and momentum."
        case .longTermInvestor:
            return "Multi-month holdings based on fundamentals and major trends."
        }
    }

    var holdingPeriod: String {
        switch self {
        case .dayTrader: return "Minutes to hours"
        case .swingTrader: return "2 days to 3 weeks"
        case .positionTrader: return "1 week to 3 months"
        case .longTermInvestor: return "1 month to 2 years"
        }
    }

    var confidenceThreshold: Int {
        switch self {
        case .dayTrader: return 70
        case .swingTrader: return 65
        case .positionTrader: return 65
        case .longTermInvestor: return 60
        }
    }

    var accentColor: Color {
        switch self {
        case .dayTrader: return .orange
        case .swingTrader: return .blue
        case .positionTrader: return .purple
        case .longTermInvestor: return .green
        }
    }

    var gradientColors: [Color] {
        switch self {
        case .dayTrader:
            return [Color.orange.opacity(0.8), Color.red.opacity(0.6)]
        case .swingTrader:
            return [Color.blue.opacity(0.8), Color.cyan.opacity(0.6)]
        case .positionTrader:
            return [Color.purple.opacity(0.8), Color.indigo.opacity(0.6)]
        case .longTermInvestor:
            return [Color.green.opacity(0.8), Color.mint.opacity(0.6)]
        }
    }
}
