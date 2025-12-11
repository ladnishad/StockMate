import SwiftUI

// MARK: - Color Extensions

extension Color {
    /// Create color from hex string
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// MARK: - View Extensions

extension View {
    /// Apply a shimmer loading effect
    @ViewBuilder
    func shimmer(isActive: Bool = true) -> some View {
        if isActive {
            self.modifier(ShimmerModifier())
        } else {
            self
        }
    }

    /// Conditionally apply a modifier
    @ViewBuilder
    func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }

    /// Read the view's size
    func readSize(onChange: @escaping (CGSize) -> Void) -> some View {
        background(
            GeometryReader { geometry in
                Color.clear
                    .preference(key: SizePreferenceKey.self, value: geometry.size)
            }
        )
        .onPreferenceChange(SizePreferenceKey.self, perform: onChange)
    }
}

// MARK: - Shimmer Modifier

struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(
                    colors: [
                        .clear,
                        .white.opacity(0.4),
                        .clear
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(width: 100)
                .offset(x: phase)
                .mask(content)
            )
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    phase = 300
                }
            }
    }
}

// MARK: - Preference Keys

struct SizePreferenceKey: PreferenceKey {
    static var defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
        value = nextValue()
    }
}

// MARK: - Double Extensions

extension Double {
    /// Format as currency string
    var asCurrency: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.currencyCode = "USD"
        formatter.maximumFractionDigits = 2
        formatter.minimumFractionDigits = 2
        return formatter.string(from: NSNumber(value: self)) ?? "$0.00"
    }

    /// Format as percentage string
    var asPercentage: String {
        String(format: "%.2f%%", self)
    }

    /// Format with sign prefix
    var withSign: String {
        self >= 0 ? "+\(String(format: "%.2f", self))" : String(format: "%.2f", self)
    }
}

// MARK: - String Extensions

extension String {
    /// Truncate string to max length with ellipsis
    func truncated(to length: Int) -> String {
        if self.count > length {
            return String(self.prefix(length)) + "..."
        }
        return self
    }
}

// MARK: - Date Extensions

extension Date {
    /// Format as relative time string (e.g., "5 min ago")
    var relativeTime: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: self, relativeTo: Date())
    }

    /// Format for market hours display
    var marketTimeFormat: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        formatter.timeZone = TimeZone(identifier: "America/New_York")
        return formatter.string(from: self)
    }
}

// MARK: - Animation Extensions

extension Animation {
    /// Smooth spring animation for UI transitions
    static var smoothSpring: Animation {
        .spring(response: 0.35, dampingFraction: 0.7)
    }

    /// Bouncy spring for emphasis
    static var bouncySpring: Animation {
        .spring(response: 0.4, dampingFraction: 0.6)
    }
}
