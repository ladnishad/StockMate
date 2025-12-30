import SwiftUI

/// Displays the smart watchlist of top contender stocks
/// The agent automatically determines the optimal trade style for each stock
struct WatchlistView: View {
    let stocks: [Stock]
    let isLoading: Bool
    let onStockTap: ((Stock) -> Void)?

    @State private var hasAppeared = false

    init(
        stocks: [Stock],
        isLoading: Bool = false,
        onStockTap: ((Stock) -> Void)? = nil
    ) {
        self.stocks = stocks
        self.isLoading = isLoading
        self.onStockTap = onStockTap
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Section header
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Top Contenders")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                        .tracking(0.5)

                    Text("Expert Picks")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.blue)
                }

                Spacer()

                // Stock count badge
                if !stocks.isEmpty && !isLoading {
                    HStack(spacing: 4) {
                        Image(systemName: "star.fill")
                            .font(.system(size: 10, weight: .semibold))

                        Text("\(stocks.count) stocks")
                            .font(.system(size: 11, weight: .semibold))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(
                        Capsule()
                            .fill(Color(.tertiarySystemGroupedBackground))
                    )
                }
            }
            .padding(.horizontal, 20)

            // Content
            if isLoading && stocks.isEmpty {
                // Loading skeleton
                VStack(spacing: 10) {
                    ForEach(0..<5, id: \.self) { _ in
                        StockRowSkeleton()
                    }
                }
                .padding(.horizontal, 20)
            } else if stocks.isEmpty {
                // Empty state
                WatchlistEmptyState()
                    .padding(.horizontal, 20)
            } else {
                // Stock list
                VStack(spacing: 10) {
                    ForEach(Array(stocks.enumerated()), id: \.element.id) { index, stock in
                        Button {
                            onStockTap?(stock)
                        } label: {
                            StockRow(stock: stock, rank: index + 1)
                        }
                        .buttonStyle(.plain)
                        .opacity(hasAppeared ? 1 : 0)
                        .offset(y: hasAppeared ? 0 : 20)
                        .animation(
                            .spring(response: 0.5, dampingFraction: 0.7)
                            .delay(Double(index) * 0.05),
                            value: hasAppeared
                        )
                    }
                }
                .padding(.horizontal, 20)
            }
        }
        .onAppear {
            withAnimation {
                hasAppeared = true
            }
        }
    }
}

/// Empty state specific to watchlist
struct WatchlistEmptyState: View {
    var body: some View {
        VStack(spacing: 16) {
            // Icon with gradient background
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.blue.opacity(0.2), Color.purple.opacity(0.2)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 72, height: 72)

                Image(systemName: "chart.line.uptrend.xyaxis")
                    .font(.system(size: 32, weight: .medium))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }

            VStack(spacing: 6) {
                Text("No Stocks Found")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text("No stocks currently meet the expert criteria. Markets may be closed or data is loading.")
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
            }

            // Criteria reminder
            HStack(spacing: 8) {
                Image(systemName: "info.circle.fill")
                    .font(.system(size: 12, weight: .medium))

                Text("Looking for high-confidence setups")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundStyle(.blue)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                Capsule()
                    .fill(Color.blue.opacity(0.1))
            )
        }
        .padding(.vertical, 32)
        .padding(.horizontal, 24)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

// MARK: - Preview

#Preview("Watchlist with Stocks") {
    ScrollView {
        WatchlistView(
            stocks: Stock.samples,
            isLoading: false
        )
    }
    .background(Color(.systemGroupedBackground))
}

#Preview("Watchlist Loading") {
    ScrollView {
        WatchlistView(
            stocks: [],
            isLoading: true
        )
    }
    .background(Color(.systemGroupedBackground))
}

#Preview("Watchlist Empty") {
    ScrollView {
        WatchlistView(
            stocks: [],
            isLoading: false
        )
    }
    .background(Color(.systemGroupedBackground))
}
