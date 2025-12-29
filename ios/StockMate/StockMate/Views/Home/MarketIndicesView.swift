import SwiftUI

/// Displays market indices in a horizontal scrolling row
struct MarketIndicesView: View {
    let indices: [MarketIndex]
    let isLoading: Bool
    let marketDirection: MarketDirection
    let isMarketOpen: Bool

    @State private var hasAppeared = false

    init(
        indices: [MarketIndex],
        isLoading: Bool = false,
        marketDirection: MarketDirection = .mixed,
        isMarketOpen: Bool = true
    ) {
        self.indices = indices
        self.isLoading = isLoading
        self.marketDirection = marketDirection
        self.isMarketOpen = isMarketOpen
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Section header
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Market Overview")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)
                        .tracking(0.5)

                    // Market direction badge
                    HStack(spacing: 5) {
                        Image(systemName: marketDirection.icon)
                            .font(.system(size: 11, weight: .bold))

                        Text(marketDirection.rawValue)
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundStyle(marketDirection.color)
                }

                Spacer()

                // Market status indicator
                HStack(spacing: 5) {
                    Circle()
                        .fill(isMarketOpen ? .green : .red)
                        .frame(width: 6, height: 6)
                        .shadow(color: (isMarketOpen ? Color.green : Color.red).opacity(0.5), radius: 3)

                    Text(isMarketOpen ? "OPEN" : "CLOSED")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(isMarketOpen ? .green : .red)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill((isMarketOpen ? Color.green : Color.red).opacity(0.1))
                )
            }
            .padding(.horizontal, 20)

            // Index cards
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    if isLoading && indices.isEmpty {
                        // Skeleton loading
                        ForEach(0..<4, id: \.self) { index in
                            IndexCardSkeleton()
                                .transition(.opacity.combined(with: .scale(scale: 0.95)))
                        }
                    } else {
                        // Actual indices
                        ForEach(Array(indices.enumerated()), id: \.element.id) { index, marketIndex in
                            IndexCard(index: marketIndex)
                                .transition(.asymmetric(
                                    insertion: .opacity.combined(with: .scale(scale: 0.9)).combined(with: .offset(y: 20)),
                                    removal: .opacity
                                ))
                                .opacity(hasAppeared ? 1 : 0)
                                .offset(y: hasAppeared ? 0 : 20)
                                .animation(
                                    .spring(response: 0.5, dampingFraction: 0.7)
                                    .delay(Double(index) * 0.08),
                                    value: hasAppeared
                                )
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 4)
            }
            .scrollClipDisabled()
        }
        .onAppear {
            withAnimation {
                hasAppeared = true
            }
        }
    }
}

// MARK: - Preview

#Preview("Market Open") {
    VStack {
        MarketIndicesView(
            indices: MarketIndex.samples,
            isLoading: false,
            marketDirection: .bullish,
            isMarketOpen: true
        )
    }
    .padding(.vertical)
    .background(Color(.systemGroupedBackground))
}

#Preview("Market Closed") {
    VStack {
        MarketIndicesView(
            indices: MarketIndex.samples,
            isLoading: false,
            marketDirection: .mixed,
            isMarketOpen: false
        )
    }
    .padding(.vertical)
    .background(Color(.systemGroupedBackground))
}

#Preview("Loading") {
    VStack {
        MarketIndicesView(
            indices: [],
            isLoading: true,
            marketDirection: .mixed,
            isMarketOpen: true
        )
    }
    .padding(.vertical)
    .background(Color(.systemGroupedBackground))
}
