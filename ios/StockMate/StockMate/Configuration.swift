import Foundation

/// App configuration that varies by build environment
/// - Debug builds (including Staging scheme): Use staging API
/// - Release builds (Production scheme): Use production API
enum AppConfiguration {

    #if DEBUG
    /// Staging environment - used for Debug builds
    static let apiBaseURL = "https://stockmate-dev.onrender.com"
    static let environment = "staging"
    #else
    /// Production environment - used for Release builds
    static let apiBaseURL = "https://stockmate-fggr.onrender.com"
    static let environment = "production"
    #endif

    /// Check if running in staging/debug mode
    static var isStaging: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }
}
