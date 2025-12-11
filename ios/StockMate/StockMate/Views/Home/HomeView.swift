import SwiftUI

/// Main home screen combining market overview, profile selection, and watchlist
struct HomeView: View {
    @StateObject private var viewModel = HomeViewModel()
    @State private var showingStockDetail = false
    @State private var selectedStock: Stock?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Market Overview Section
                    MarketIndicesView(
                        indices: viewModel.indices,
                        isLoading: viewModel.isLoadingIndices,
                        marketDirection: viewModel.marketDirection
                    )

                    // Profile Selector Section
                    ProfileSelectorView(selected: $viewModel.selectedProfile)

                    // Smart Watchlist Section
                    WatchlistView(
                        stocks: viewModel.watchlistStocks,
                        isLoading: viewModel.isLoadingWatchlist,
                        profile: viewModel.selectedProfile
                    ) { stock in
                        selectedStock = stock
                        showingStockDetail = true
                    }

                    // Bottom padding for scroll
                    Spacer()
                        .frame(height: 20)
                }
                .padding(.top, 8)
            }
            .scrollIndicators(.hidden)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("StockMate")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    RefreshButton(isRefreshing: viewModel.isRefreshing) {
                        Task {
                            await viewModel.refresh()
                        }
                    }
                }
            }
            .refreshable {
                await viewModel.refresh()
            }
            .sheet(isPresented: $showingStockDetail) {
                if let stock = selectedStock {
                    StockDetailSheet(stock: stock, profile: viewModel.selectedProfile)
                }
            }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) {
                Button("Retry") {
                    Task {
                        await viewModel.refresh()
                    }
                }
                Button("Dismiss", role: .cancel) {
                    viewModel.dismissError()
                }
            } message: {
                if let error = viewModel.error {
                    Text(error)
                }
            }
        }
        .task {
            await viewModel.loadInitialData()
        }
    }
}

/// Toolbar refresh button with loading state
struct RefreshButton: View {
    let isRefreshing: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            if isRefreshing {
                ProgressView()
                    .scaleEffect(0.8)
            } else {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 15, weight: .semibold))
            }
        }
        .disabled(isRefreshing)
    }
}

/// Sheet showing stock details
struct StockDetailSheet: View {
    let stock: Stock
    let profile: TraderProfile
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Header with score
                    StockDetailHeader(stock: stock)

                    // Trade Plan
                    if let tradePlan = stock.tradePlan {
                        TradePlanCard(tradePlan: tradePlan, profile: profile)
                    }

                    // Reasons
                    ReasonsCard(reasons: stock.reasons)

                    // Risk Warning
                    RiskDisclaimer()
                }
                .padding(20)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle(stock.symbol)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
}

/// Stock detail header with large score
struct StockDetailHeader: View {
    let stock: Stock

    private var scoreColor: Color {
        if stock.score >= 75 { return .green }
        if stock.score >= 65 { return .blue }
        if stock.score >= 50 { return .orange }
        return .red
    }

    var body: some View {
        VStack(spacing: 16) {
            // Large score circle
            ZStack {
                Circle()
                    .stroke(scoreColor.opacity(0.2), lineWidth: 8)
                    .frame(width: 100, height: 100)

                Circle()
                    .trim(from: 0, to: stock.score / 100)
                    .stroke(
                        scoreColor,
                        style: StrokeStyle(lineWidth: 8, lineCap: .round)
                    )
                    .frame(width: 100, height: 100)
                    .rotationEffect(.degrees(-90))

                VStack(spacing: 2) {
                    Text(String(format: "%.0f", stock.score))
                        .font(.system(size: 32, weight: .bold, design: .rounded))
                        .foregroundStyle(scoreColor)

                    Text("SCORE")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.secondary)
                }
            }

            // Recommendation
            HStack(spacing: 8) {
                Image(systemName: stock.recommendation.uppercased() == "BUY" ? "checkmark.seal.fill" : "minus.circle.fill")
                    .font(.system(size: 18, weight: .bold))

                Text(stock.recommendation.uppercased() == "BUY" ? "Strong Buy Signal" : "Hold / Wait")
                    .font(.system(size: 16, weight: .semibold))
            }
            .foregroundStyle(stock.recommendation.uppercased() == "BUY" ? .green : .secondary)

            // Price
            Text(stock.formattedPrice)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
        }
        .padding(24)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

/// Trade plan card with entry, stop, targets
struct TradePlanCard: View {
    let tradePlan: TradePlan
    let profile: TraderProfile

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack {
                Image(systemName: "target")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(profile.accentColor)

                Text("Trade Plan")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.primary)
            }

            // Entry
            TradePlanRow(
                label: "Entry Zone",
                value: tradePlan.formattedEntry,
                icon: "arrow.right.circle.fill",
                color: .blue
            )

            // Stop Loss
            TradePlanRow(
                label: "Stop Loss",
                value: tradePlan.formattedStopLoss,
                icon: "xmark.octagon.fill",
                color: .red
            )

            // Targets
            VStack(alignment: .leading, spacing: 8) {
                Text("Targets")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)

                ForEach(Array(tradePlan.targets.enumerated()), id: \.offset) { index, target in
                    TradePlanRow(
                        label: "Target \(index + 1)",
                        value: String(format: "$%.2f", target),
                        icon: "flag.fill",
                        color: .green
                    )
                }
            }

            // Risk/Reward
            if let riskReward = tradePlan.riskRewardRatio {
                HStack {
                    Text("Risk/Reward")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text("1:\(String(format: "%.1f", riskReward))")
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundStyle(riskReward >= 2 ? .green : .orange)
                }
                .padding(.top, 4)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

/// Single row in trade plan
struct TradePlanRow: View {
    let label: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        HStack {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(color)

                Text(label)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(value)
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .foregroundStyle(.primary)
        }
    }
}

/// Card showing analysis reasons
struct ReasonsCard: View {
    let reasons: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.yellow)

                Text("Analysis Highlights")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.primary)
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(reasons, id: \.self) { reason in
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(Color.accentColor)
                            .frame(width: 6, height: 6)
                            .padding(.top, 6)

                        Text(reason)
                            .font(.system(size: 14, weight: .regular))
                            .foregroundStyle(.primary)
                    }
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

/// Risk disclaimer footer
struct RiskDisclaimer: View {
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(.orange)

            Text("For educational purposes only. Not financial advice. Always do your own research.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.orange.opacity(0.08))
        )
    }
}

// MARK: - Preview

#Preview("Home View") {
    HomeView()
}
