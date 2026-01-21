import Foundation

/// App configuration that varies by build environment
/// - Local builds: Use local backend (run ./scripts/run-local.sh)
/// - Debug/Staging builds: Use staging API
/// - Release/Production builds: Use production API
enum AppConfiguration {

    enum Environment: String {
        case local
        case staging
        case production
    }

    /// Current environment based on build configuration
    static var environment: Environment {
        #if LOCAL
        return .local
        #elseif DEBUG
        return .staging
        #else
        return .production
        #endif
    }

    /// Base URL for API requests
    static var apiBaseURL: String {
        switch environment {
        case .local:
            // Hardcoded local IP - update if your network changes
            // Run ./scripts/run-local.sh to see your current IP
            return "http://192.168.1.66:8000"
        case .staging:
            return "https://stockmate-dev.onrender.com"
        case .production:
            return "https://stockmate-fggr.onrender.com"
        }
    }

    /// Check if running in local mode
    static var isLocal: Bool {
        environment == .local
    }

    /// Check if running in staging/debug mode
    static var isStaging: Bool {
        environment == .staging
    }

    /// Check if running in production mode
    static var isProduction: Bool {
        environment == .production
    }
}
