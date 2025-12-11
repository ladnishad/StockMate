import SwiftUI

/// Displays market indices in a horizontal scrolling row
struct MarketIndicesView: View {
    let indices: [MarketIndex]
    let isLoading: Bool
    let marketDirection: MarketDirection

    @State private var hasAppeared = false

    init(
        indices: [MarketIndex],
        isLoading: Bool = false,
        marketDirection: MarketDirection = .mixed
    ) {
        self.indices = indices
        self.isLoading = isLoading
        self.marketDirection = marketDirection
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

                // Live indicator
                HStack(spacing: 5) {
                    Circle()
                        .fill(.green)
                        .frame(width: 6, height: 6)
                        .shadow(color: .green.opacity(0.5), radius: 3)

                    Text("LIVE")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.green)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(Color.green.opacity(0.1))
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

#Preview("Market Indices") {
    VStack {
        MarketIndicesView(
            indices: MarketIndex.samples,
            isLoading: false,
            marketDirection: .bullish
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
            marketDirection: .mixed
        )
    }
    .padding(.vertical)
    .background(Color(.systemGroupedBackground))
}
