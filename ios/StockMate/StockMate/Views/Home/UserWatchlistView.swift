import SwiftUI

/// Displays the user's watchlist with swipe-to-delete and navigation to detail
struct UserWatchlistView: View {
    let items: [WatchlistItem]
    let isLoading: Bool
    let onDelete: ((String) -> Void)?

    @State private var hasAppeared = false

    init(
        items: [WatchlistItem],
        isLoading: Bool = false,
        onDelete: ((String) -> Void)? = nil
    ) {
        self.items = items
        self.isLoading = isLoading
        self.onDelete = onDelete
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Section header
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("My Watchlist")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                        .tracking(0.5)

                    Text("Tickers you're tracking")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.tertiary)
                }

                Spacer()

                // Stock count badge
                if !items.isEmpty && !isLoading {
                    HStack(spacing: 4) {
                        Image(systemName: "eye.fill")
                            .font(.system(size: 10, weight: .semibold))

                        Text("\(items.count) \(items.count == 1 ? "stock" : "stocks")")
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
            if isLoading && items.isEmpty {
                // Loading skeleton
                VStack(spacing: 10) {
                    ForEach(0..<3, id: \.self) { _ in
                        WatchlistItemSkeleton()
                    }
                }
                .padding(.horizontal, 20)
            } else if items.isEmpty {
                // Empty state
                UserWatchlistEmptyState()
                    .padding(.horizontal, 20)
            } else {
                // Watchlist items
                VStack(spacing: 10) {
                    ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
                        NavigationLink(value: item.symbol) {
                            UserWatchlistRow(item: item)
                        }
                        .buttonStyle(.plain)
                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                            Button(role: .destructive) {
                                onDelete?(item.symbol)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
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

/// Single row displaying a watchlist item - redesigned for more information
struct UserWatchlistRow: View {
    let item: WatchlistItem

    private var changeColor: Color {
        guard let change = item.change else { return .secondary }
        return change >= 0 ? Color(.systemGreen) : Color(.systemRed)
    }

    var body: some View {
        VStack(spacing: 12) {
            // Top row: Symbol + Price/Change
            HStack(alignment: .top) {
                // Symbol
                Text(item.symbol)
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)

                Spacer()

                // Price and changes
                VStack(alignment: .trailing, spacing: 3) {
                    if let price = item.currentPrice {
                        Text(formatPrice(price))
                            .font(.system(size: 17, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.primary)
                    } else {
                        Text("--")
                            .font(.system(size: 17, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }

                    // Dollar change + Percent change
                    HStack(spacing: 4) {
                        if let change = item.change {
                            Text(String(format: "%+.2f", change))
                                .font(.system(size: 13, weight: .medium, design: .monospaced))
                        }

                        if let changePct = item.changePct {
                            Text("(\(String(format: "%+.2f%%", changePct)))")
                                .font(.system(size: 13, weight: .medium, design: .monospaced))
                        }
                    }
                    .foregroundStyle(changeColor)
                }
            }

            // Bottom row: Score ring + Badge + Alert + Chevron
            HStack(spacing: 12) {
                // Mini score ring
                if let score = item.score {
                    MiniScoreRing(score: score)
                }

                // Recommendation badge
                if let recommendation = item.recommendation {
                    Text(recommendation)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(recommendation.uppercased() == "BUY" ? Color(.systemGreen) : .secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            Capsule()
                                .fill(recommendation.uppercased() == "BUY" ? Color(.systemGreen).opacity(0.12) : Color(.tertiarySystemGroupedBackground))
                        )
                }

                // Alert indicator
                if item.alertsEnabled {
                    Image(systemName: "bell.fill")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.orange)
                }

                Spacer()

                // Chevron
                Image(systemName: "chevron.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }

    private func formatPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.0f", price)
        } else {
            return String(format: "$%.2f", price)
        }
    }
}

/// Mini circular score ring for watchlist cards
private struct MiniScoreRing: View {
    let score: Double

    var body: some View {
        ZStack {
            // Background track
            Circle()
                .stroke(Color(.systemGray5), lineWidth: 3)

            // Filled arc
            Circle()
                .trim(from: 0, to: score / 100)
                .stroke(
                    Color(.systemGray2),
                    style: StrokeStyle(lineWidth: 3, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))

            // Score text
            Text("\(Int(score))")
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
        }
        .frame(width: 32, height: 32)
    }
}

/// Loading skeleton for watchlist item
struct WatchlistItemSkeleton: View {
    @State private var isAnimating = false

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(.systemGray4))
                    .frame(width: 50, height: 18)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(.systemGray5))
                    .frame(width: 30, height: 12)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 6) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(.systemGray4))
                    .frame(width: 70, height: 18)

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color(.systemGray5))
                    .frame(width: 50, height: 12)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
        .opacity(isAnimating ? 0.5 : 1.0)
        .animation(
            .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
            value: isAnimating
        )
        .onAppear {
            isAnimating = true
        }
    }
}

/// Empty state for user watchlist
struct UserWatchlistEmptyState: View {
    var body: some View {
        VStack(spacing: 16) {
            // Icon
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.12))
                    .frame(width: 72, height: 72)

                Image(systemName: "plus.circle.dashed")
                    .font(.system(size: 32, weight: .medium))
                    .foregroundStyle(Color.accentColor)
            }

            VStack(spacing: 6) {
                Text("No Stocks Yet")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text("Add tickers to start tracking stocks.\nUse the search bar or + button to add your first stock.")
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
            }

            // Hint
            HStack(spacing: 6) {
                Image(systemName: "lightbulb.fill")
                    .font(.system(size: 11, weight: .medium))

                Text("Tip: Each stock will be monitored by AI")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundStyle(.orange)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                Capsule()
                    .fill(Color.orange.opacity(0.1))
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

#Preview("User Watchlist with Items") {
    NavigationStack {
        ScrollView {
            UserWatchlistView(
                items: WatchlistItem.samples,
                isLoading: false
            ) { symbol in
                print("Delete: \(symbol)")
            }
        }
        .background(Color(.systemGroupedBackground))
    }
}

#Preview("User Watchlist Loading") {
    ScrollView {
        UserWatchlistView(
            items: [],
            isLoading: true
        )
    }
    .background(Color(.systemGroupedBackground))
}

#Preview("User Watchlist Empty") {
    ScrollView {
        UserWatchlistView(
            items: [],
            isLoading: false
        )
    }
    .background(Color(.systemGroupedBackground))
}
