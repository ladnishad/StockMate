import SwiftUI

/// Tappable search bar that opens the ticker search sheet
struct SearchBarView: View {
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.secondary)

                Text("Search ticker or company...")
                    .font(.system(size: 16, weight: .regular))
                    .foregroundStyle(.tertiary)

                Spacer()
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Search for stocks")
        .accessibilityHint("Opens search to find and add stocks to your watchlist")
    }
}

// MARK: - Preview

#Preview("Search Bar") {
    VStack {
        SearchBarView {
            print("Search tapped")
        }
        .padding()

        Spacer()
    }
    .background(Color(.systemGroupedBackground))
}
