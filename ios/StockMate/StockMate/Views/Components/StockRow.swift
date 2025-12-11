import SwiftUI

/// A row displaying a stock with score, recommendation, and key info
struct StockRow: View {
    let stock: Stock
    let rank: Int?

    @State private var isPressed = false

    init(stock: Stock, rank: Int? = nil) {
        self.stock = stock
        self.rank = rank
    }

    var body: some View {
        HStack(spacing: 14) {
            // Rank badge (optional)
            if let rank {
                RankBadge(rank: rank, isTopThree: rank <= 3)
            }

            // Stock info
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    // Symbol
                    Text(stock.symbol)
                        .font(.system(size: 17, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)

                    // Sector tag
                    if let sector = stock.sectorName {
                        Text(sector)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                Capsule()
                                    .fill(Color(.systemGray5))
                            )
                    }
                }

                // Top reason
                if let reason = stock.reasons.first {
                    Text(reason)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            // Right side: Price & Score
            VStack(alignment: .trailing, spacing: 4) {
                // Price
                Text(stock.formattedPrice)
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                // Score & Recommendation
                HStack(spacing: 6) {
                    ScoreBadge(score: stock.score)
                    RecommendationTag(recommendation: stock.recommendation)
                }
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
        .scaleEffect(isPressed ? 0.98 : 1.0)
        .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isPressed)
    }
}

// MARK: - Supporting Components

/// Displays the stock's rank in the watchlist
struct RankBadge: View {
    let rank: Int
    let isTopThree: Bool

    var body: some View {
        ZStack {
            if isTopThree {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: gradientColors,
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 28, height: 28)
            } else {
                Circle()
                    .fill(Color(.systemGray5))
                    .frame(width: 28, height: 28)
            }

            Text("\(rank)")
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundStyle(isTopThree ? .white : .secondary)
        }
    }

    private var gradientColors: [Color] {
        switch rank {
        case 1: return [Color.yellow, Color.orange]
        case 2: return [Color.gray.opacity(0.8), Color.gray.opacity(0.5)]
        case 3: return [Color.orange.opacity(0.8), Color.brown.opacity(0.6)]
        default: return [Color.gray, Color.gray]
        }
    }
}

/// Displays the analysis score with color coding
struct ScoreBadge: View {
    let score: Double

    var body: some View {
        Text(String(format: "%.0f", score))
            .font(.system(size: 13, weight: .bold, design: .rounded))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(backgroundColor)
            )
    }

    private var backgroundColor: Color {
        if score >= 75 { return .green }
        if score >= 65 { return .blue }
        if score >= 50 { return .orange }
        return .red
    }
}

/// Displays BUY or NO_BUY recommendation
struct RecommendationTag: View {
    let recommendation: String

    private var isBuy: Bool {
        recommendation.uppercased() == "BUY"
    }

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: isBuy ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.system(size: 10, weight: .bold))

            Text(isBuy ? "BUY" : "HOLD")
                .font(.system(size: 10, weight: .bold))
        }
        .foregroundStyle(isBuy ? .green : .secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(isBuy ? Color.green.opacity(0.12) : Color(.systemGray5))
        )
    }
}

/// Skeleton loading state for StockRow
struct StockRowSkeleton: View {
    @State private var isAnimating = false

    var body: some View {
        HStack(spacing: 14) {
            // Rank skeleton
            Circle()
                .fill(.quaternary)
                .frame(width: 28, height: 28)

            // Info skeleton
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(.quaternary)
                        .frame(width: 50, height: 16)

                    RoundedRectangle(cornerRadius: 8)
                        .fill(.quaternary)
                        .frame(width: 60, height: 16)
                }

                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                    .frame(width: 120, height: 12)
            }

            Spacer()

            // Price & score skeleton
            VStack(alignment: .trailing, spacing: 6) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                    .frame(width: 60, height: 16)

                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(.quaternary)
                        .frame(width: 30, height: 22)

                    RoundedRectangle(cornerRadius: 5)
                        .fill(.quaternary)
                        .frame(width: 40, height: 22)
                }
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
        .opacity(isAnimating ? 0.6 : 1.0)
        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)
        .onAppear { isAnimating = true }
    }
}

// MARK: - Preview

#Preview("Stock Rows") {
    VStack(spacing: 10) {
        ForEach(Array(Stock.samples.enumerated()), id: \.element.id) { index, stock in
            StockRow(stock: stock, rank: index + 1)
        }
    }
    .padding()
    .background(Color(.systemGroupedBackground))
}

#Preview("Skeleton") {
    VStack(spacing: 10) {
        StockRowSkeleton()
        StockRowSkeleton()
        StockRowSkeleton()
    }
    .padding()
    .background(Color(.systemGroupedBackground))
}
