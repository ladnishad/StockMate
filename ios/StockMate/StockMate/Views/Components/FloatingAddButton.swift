import SwiftUI

/// Floating action button for adding tickers to watchlist
struct FloatingAddButton: View {
    let action: () -> Void

    @State private var isPressed = false

    var body: some View {
        Button(action: {
            // Haptic feedback
            let generator = UIImpactFeedbackGenerator(style: .medium)
            generator.impactOccurred()

            action()
        }) {
            ZStack {
                // Shadow layer
                Circle()
                    .fill(Color.accentColor)
                    .frame(width: 56, height: 56)
                    .shadow(color: Color.accentColor.opacity(0.4), radius: 8, x: 0, y: 4)

                // Icon
                Image(systemName: "plus")
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(.white)
            }
        }
        .buttonStyle(FABButtonStyle())
        .accessibilityLabel("Add ticker")
        .accessibilityHint("Opens search to add a new stock to your watchlist")
    }
}

/// Custom button style for FAB with press animation
struct FABButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.9 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.6), value: configuration.isPressed)
    }
}

/// Container view that positions the FAB in bottom-right corner
struct FloatingAddButtonContainer<Content: View>: View {
    @ViewBuilder let content: Content
    let onAdd: () -> Void

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            content

            FloatingAddButton(action: onAdd)
                .padding(.trailing, 20)
                .padding(.bottom, 20)
        }
    }
}

// MARK: - Preview

#Preview("Floating Add Button") {
    ZStack {
        Color(.systemGroupedBackground)
            .ignoresSafeArea()

        VStack {
            Text("Your content here")
                .foregroundStyle(.secondary)
        }

        VStack {
            Spacer()
            HStack {
                Spacer()
                FloatingAddButton {
                    print("Add tapped")
                }
                .padding(20)
            }
        }
    }
}

#Preview("With Container") {
    FloatingAddButtonContainer(
        content: {
            ScrollView {
                VStack(spacing: 20) {
                    ForEach(0..<10, id: \.self) { i in
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.secondarySystemGroupedBackground))
                            .frame(height: 80)
                            .padding(.horizontal)
                    }
                }
            }
        },
        onAdd: { print("Add tapped") }
    )
    .background(Color(.systemGroupedBackground))
}
