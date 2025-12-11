import SwiftUI

/// A sleek card displaying a market index with price and change
struct IndexCard: View {
    let index: MarketIndex
    let isCompact: Bool

    @State private var isPressed = false

    init(index: MarketIndex, isCompact: Bool = false) {
        self.index = index
        self.isCompact = isCompact
    }

    var body: some View {
        VStack(alignment: .leading, spacing: isCompact ? 6 : 10) {
            // Header: Symbol & Direction
            HStack(spacing: 6) {
                Text(index.symbol)
                    .font(.system(size: isCompact ? 13 : 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)

                Spacer()

                // Direction indicator
                Image(systemName: index.isUp ? "arrow.up.right" : "arrow.down.right")
                    .font(.system(size: isCompact ? 10 : 12, weight: .bold))
                    .foregroundStyle(index.isUp ? .green : .red)
            }

            // Price
            Text(index.formattedPrice)
                .font(.system(size: isCompact ? 20 : 24, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
                .contentTransition(.numericText())

            // Change badge
            HStack(spacing: 4) {
                Text(index.formattedChange)
                    .font(.system(size: isCompact ? 11 : 13, weight: .medium, design: .monospaced))

                Text(index.formattedChangePct)
                    .font(.system(size: isCompact ? 11 : 13, weight: .semibold, design: .monospaced))
            }
            .foregroundStyle(index.isUp ? .green : .red)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(index.isUp ? Color.green.opacity(0.12) : Color.red.opacity(0.12))
            )

            // Index name
            Text(index.name)
                .font(.system(size: isCompact ? 10 : 11, weight: .medium))
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .padding(isCompact ? 12 : 16)
        .frame(width: isCompact ? 140 : 160)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.ultraThinMaterial)
                .shadow(color: .black.opacity(0.06), radius: 8, x: 0, y: 4)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(
                    LinearGradient(
                        colors: [
                            index.isUp ? Color.green.opacity(0.3) : Color.red.opacity(0.3),
                            Color.clear
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        )
        .scaleEffect(isPressed ? 0.96 : 1.0)
        .animation(.spring(response: 0.3, dampingFraction: 0.6), value: isPressed)
    }
}

/// A skeleton loading placeholder for IndexCard
struct IndexCardSkeleton: View {
    @State private var isAnimating = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header skeleton
            HStack {
                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                    .frame(width: 40, height: 14)
                Spacer()
                Circle()
                    .fill(.quaternary)
                    .frame(width: 14, height: 14)
            }

            // Price skeleton
            RoundedRectangle(cornerRadius: 4)
                .fill(.quaternary)
                .frame(width: 90, height: 24)

            // Change skeleton
            RoundedRectangle(cornerRadius: 6)
                .fill(.quaternary)
                .frame(width: 80, height: 24)

            // Name skeleton
            RoundedRectangle(cornerRadius: 4)
                .fill(.quaternary)
                .frame(width: 60, height: 10)
        }
        .padding(16)
        .frame(width: 160)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(.ultraThinMaterial)
        )
        .opacity(isAnimating ? 0.6 : 1.0)
        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)
        .onAppear { isAnimating = true }
    }
}

// MARK: - Preview

#Preview("Index Cards") {
    ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 12) {
            ForEach(MarketIndex.samples) { index in
                IndexCard(index: index)
            }
        }
        .padding()
    }
    .background(Color(.systemGroupedBackground))
}

#Preview("Skeleton") {
    HStack(spacing: 12) {
        IndexCardSkeleton()
        IndexCardSkeleton()
    }
    .padding()
    .background(Color(.systemGroupedBackground))
}
