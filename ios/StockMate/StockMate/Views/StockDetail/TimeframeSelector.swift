import SwiftUI

/// Segmented picker for chart timeframe selection
struct TimeframeSelector: View {
    @Binding var selected: ChartTimeframe

    var body: some View {
        HStack(spacing: 0) {
            ForEach(ChartTimeframe.allCases) { timeframe in
                TimeframeButton(
                    timeframe: timeframe,
                    isSelected: timeframe == selected
                ) {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        selected = timeframe
                    }
                    // Haptic feedback
                    let generator = UISelectionFeedbackGenerator()
                    generator.selectionChanged()
                }
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

/// Individual timeframe button
struct TimeframeButton: View {
    let timeframe: ChartTimeframe
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(timeframe.rawValue)
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundStyle(isSelected ? .white : .secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .background {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color.accentColor)
                    }
                }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Preview

#Preview {
    struct PreviewWrapper: View {
        @State private var selected: ChartTimeframe = .oneMonth

        var body: some View {
            VStack(spacing: 20) {
                TimeframeSelector(selected: $selected)

                Text("Selected: \(selected.displayName)")
                    .foregroundStyle(.secondary)
            }
            .padding()
        }
    }

    return PreviewWrapper()
}
