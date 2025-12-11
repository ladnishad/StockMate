import SwiftUI

/// A beautiful profile selector with animated pills
struct ProfileSelectorView: View {
    @Binding var selected: TraderProfile
    @Namespace private var animation

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Section header
            HStack {
                Text("Trading Style")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)

                Spacer()

                // Selected profile info
                Text(selected.holdingPeriod)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 20)

            // Profile pills
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(TraderProfile.allCases) { profile in
                        ProfilePill(
                            profile: profile,
                            isSelected: profile == selected,
                            animation: animation
                        ) {
                            withAnimation(.spring(response: 0.35, dampingFraction: 0.7)) {
                                selected = profile
                            }
                            // Haptic feedback
                            let generator = UIImpactFeedbackGenerator(style: .light)
                            generator.impactOccurred()
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 4)
            }
            .scrollClipDisabled()

            // Profile description card
            ProfileDescriptionCard(profile: selected)
                .padding(.horizontal, 20)
        }
    }
}

/// Individual profile pill button
struct ProfilePill: View {
    let profile: TraderProfile
    let isSelected: Bool
    let animation: Namespace.ID
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: profile.icon)
                    .font(.system(size: 13, weight: .semibold))

                Text(profile.displayName)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
            }
            .foregroundStyle(isSelected ? .white : .primary)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background {
                if isSelected {
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: profile.gradientColors,
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .matchedGeometryEffect(id: "pill", in: animation)
                        .shadow(color: profile.accentColor.opacity(0.3), radius: 8, y: 4)
                } else {
                    Capsule()
                        .fill(Color(.secondarySystemGroupedBackground))
                }
            }
        }
        .buttonStyle(.plain)
    }
}

/// Card showing the selected profile's description
struct ProfileDescriptionCard: View {
    let profile: TraderProfile

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            // Icon
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: profile.gradientColors,
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 44, height: 44)

                Image(systemName: profile.icon)
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(profile.fullName)
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(profile.description)
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)

                // Stats row
                HStack(spacing: 12) {
                    StatBadge(
                        icon: "clock",
                        text: profile.holdingPeriod,
                        color: profile.accentColor
                    )

                    StatBadge(
                        icon: "target",
                        text: "\(profile.confidenceThreshold)% threshold",
                        color: .orange
                    )
                }
                .padding(.top, 4)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(
                    LinearGradient(
                        colors: [
                            profile.accentColor.opacity(0.3),
                            Color.clear
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        )
        .contentShape(Rectangle())
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: profile)
    }
}

/// Small stat badge
struct StatBadge: View {
    let icon: String
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))

            Text(text)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(color)
    }
}

// MARK: - Preview

#Preview("Profile Selector") {
    struct PreviewWrapper: View {
        @State private var selected: TraderProfile = .swingTrader

        var body: some View {
            ProfileSelectorView(selected: $selected)
                .padding(.vertical)
                .background(Color(.systemGroupedBackground))
        }
    }

    return PreviewWrapper()
}
