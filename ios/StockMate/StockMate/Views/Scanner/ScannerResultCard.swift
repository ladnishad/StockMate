import SwiftUI

/// A card displaying a scanner result with confidence grade and action buttons
struct ScannerResultCard: View {
    let result: ScannerResult
    let onAdd: () -> Void
    let onTap: () -> Void

    @State private var isPressed = false
    @State private var showAddConfirmation = false

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                // Left side: Symbol, badges, description
                VStack(alignment: .leading, spacing: 6) {
                    // Top row: Symbol + status pills
                    HStack(spacing: 8) {
                        Text(result.symbol)
                            .font(.system(size: 17, weight: .bold, design: .rounded))
                            .foregroundStyle(.primary)

                        if result.isNew {
                            NewPill()
                        }

                        if result.isWatching {
                            WatchingPill()
                        }
                    }

                    // Description
                    Text(result.description)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    // Warnings (if any)
                    if !result.warnings.isEmpty {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(.orange)

                            Text(result.warnings.first ?? "")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(.orange)
                        }
                    }
                }

                Spacer()

                // Right side: Price, grade, add button
                VStack(alignment: .trailing, spacing: 8) {
                    // Price
                    Text(result.formattedPrice)
                        .font(.system(size: 17, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary)

                    // Grade badge
                    ConfidenceGradeBadge(grade: result.confidenceGrade)

                    // Add/Watching button
                    if result.isWatching {
                        ViewButton(action: onTap)
                    } else {
                        AddButton(action: {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                showAddConfirmation = true
                            }
                            onAdd()

                            // Hide confirmation after delay
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                withAnimation {
                                    showAddConfirmation = false
                                }
                            }
                        }, showConfirmation: showAddConfirmation)
                    }
                }
            }
            .padding(.vertical, 14)
            .padding(.horizontal, 16)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))
            )
            .scaleEffect(isPressed ? 0.98 : 1.0)
            .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isPressed)
        }
        .buttonStyle(PlainButtonStyle())
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in isPressed = true }
                .onEnded { _ in isPressed = false }
        )
    }
}

// MARK: - Supporting Components

/// "NEW" pill badge
struct NewPill: View {
    var body: some View {
        Text("NEW")
            .font(.system(size: 9, weight: .bold, design: .rounded))
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Color.blue, Color.blue.opacity(0.8)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            )
    }
}

/// "Watching" pill badge
struct WatchingPill: View {
    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: "eye.fill")
                .font(.system(size: 8))
            Text("Watching")
                .font(.system(size: 9, weight: .medium))
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(
            Capsule()
                .fill(Color(.systemGray5))
        )
    }
}

/// Confidence grade badge with color coding
struct ConfidenceGradeBadge: View {
    let grade: ConfidenceGrade

    var body: some View {
        Text(grade.displayText)
            .font(.system(size: 14, weight: .bold, design: .rounded))
            .foregroundStyle(.white)
            .frame(minWidth: 32)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(backgroundColor)
            )
            .shadow(color: shadowColor.opacity(0.3), radius: 4, x: 0, y: 2)
    }

    private var backgroundColor: some ShapeStyle {
        switch grade {
        case .aPlus:
            return AnyShapeStyle(
                LinearGradient(
                    colors: [Color.green, Color.green.opacity(0.85)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        case .a:
            return AnyShapeStyle(Color.green)
        case .bPlus:
            return AnyShapeStyle(Color.blue)
        case .b:
            return AnyShapeStyle(Color.orange)
        case .c:
            return AnyShapeStyle(Color.gray)
        }
    }

    private var shadowColor: Color {
        switch grade {
        case .aPlus, .a: return .green
        case .bPlus: return .blue
        case .b: return .orange
        case .c: return .gray
        }
    }
}

/// Add button with + icon
struct AddButton: View {
    let action: () -> Void
    var showConfirmation: Bool = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if showConfirmation {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                    Text("Added")
                        .font(.system(size: 11, weight: .semibold))
                } else {
                    Image(systemName: "plus")
                        .font(.system(size: 12, weight: .bold))
                    Text("Add")
                        .font(.system(size: 11, weight: .semibold))
                }
            }
            .foregroundStyle(showConfirmation ? .green : .blue)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(showConfirmation ? Color.green.opacity(0.12) : Color.blue.opacity(0.12))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

/// View button for already-watched stocks
struct ViewButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: "arrow.right")
                    .font(.system(size: 10, weight: .bold))
                Text("View")
                    .font(.system(size: 11, weight: .semibold))
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(Color(.systemGray5))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

/// Skeleton loading state for ScannerResultCard
struct ScannerResultCardSkeleton: View {
    @State private var isAnimating = false

    var body: some View {
        HStack(spacing: 14) {
            // Left side skeleton
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(.quaternary)
                        .frame(width: 55, height: 18)

                    Capsule()
                        .fill(.quaternary)
                        .frame(width: 35, height: 16)
                }

                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                    .frame(width: 200, height: 14)

                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                    .frame(width: 140, height: 12)
            }

            Spacer()

            // Right side skeleton
            VStack(alignment: .trailing, spacing: 8) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                    .frame(width: 65, height: 18)

                RoundedRectangle(cornerRadius: 8)
                    .fill(.quaternary)
                    .frame(width: 40, height: 28)

                Capsule()
                    .fill(.quaternary)
                    .frame(width: 55, height: 26)
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))
        )
        .opacity(isAnimating ? 0.6 : 1.0)
        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)
        .onAppear { isAnimating = true }
    }
}

// MARK: - Preview

#Preview("Scanner Result Cards") {
    ScrollView {
        VStack(spacing: 10) {
            ForEach(ScannerResult.samples) { result in
                ScannerResultCard(
                    result: result,
                    onAdd: { print("Add \(result.symbol)") },
                    onTap: { print("Tap \(result.symbol)") }
                )
            }
        }
        .padding()
    }
    .background(Color(.systemGroupedBackground))
}

#Preview("Skeleton") {
    VStack(spacing: 10) {
        ScannerResultCardSkeleton()
        ScannerResultCardSkeleton()
        ScannerResultCardSkeleton()
    }
    .padding()
    .background(Color(.systemGroupedBackground))
}
