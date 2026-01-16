//
//  KeychainHelper.swift
//  StockMate
//
//  Secure storage helper using iOS Keychain for auth tokens.
//

import Foundation
import Security

final class KeychainHelper {
    static let shared = KeychainHelper()
    private init() {}

    // MARK: - Keychain Keys

    private enum Keys {
        static let accessToken = "com.stockmate.accessToken"
        static let refreshToken = "com.stockmate.refreshToken"
        static let userEmail = "com.stockmate.userEmail"
        static let userId = "com.stockmate.userId"
        static let tokenExpiration = "com.stockmate.tokenExpiration"
        static let lastRefreshTime = "com.stockmate.lastRefreshTime"
    }

    // MARK: - Token Management

    var accessToken: String? {
        get { read(key: Keys.accessToken) }
        set {
            if let value = newValue {
                save(key: Keys.accessToken, value: value)
            } else {
                delete(key: Keys.accessToken)
            }
        }
    }

    var refreshToken: String? {
        get { read(key: Keys.refreshToken) }
        set {
            if let value = newValue {
                save(key: Keys.refreshToken, value: value)
            } else {
                delete(key: Keys.refreshToken)
            }
        }
    }

    var userEmail: String? {
        get { read(key: Keys.userEmail) }
        set {
            if let value = newValue {
                save(key: Keys.userEmail, value: value)
            } else {
                delete(key: Keys.userEmail)
            }
        }
    }

    var userId: String? {
        get { read(key: Keys.userId) }
        set {
            if let value = newValue {
                save(key: Keys.userId, value: value)
            } else {
                delete(key: Keys.userId)
            }
        }
    }

    var tokenExpirationDate: Date? {
        get {
            guard let timestampString = read(key: Keys.tokenExpiration),
                  let timestamp = Double(timestampString) else {
                return nil
            }
            return Date(timeIntervalSince1970: timestamp)
        }
        set {
            if let date = newValue {
                save(key: Keys.tokenExpiration, value: String(date.timeIntervalSince1970))
            } else {
                delete(key: Keys.tokenExpiration)
            }
        }
    }

    /// Tracks when the last successful token refresh occurred
    var lastRefreshTime: Date? {
        get {
            guard let timestampString = read(key: Keys.lastRefreshTime),
                  let timestamp = Double(timestampString) else {
                return nil
            }
            return Date(timeIntervalSince1970: timestamp)
        }
        set {
            if let date = newValue {
                save(key: Keys.lastRefreshTime, value: String(date.timeIntervalSince1970))
            } else {
                delete(key: Keys.lastRefreshTime)
            }
        }
    }

    // MARK: - Save Auth Session

    func saveAuthSession(response: AuthResponse) {
        accessToken = response.accessToken
        refreshToken = response.refreshToken
        userEmail = response.user.email
        userId = response.user.id
        // Calculate and store token expiration (expires_in is in seconds)
        tokenExpirationDate = Date().addingTimeInterval(TimeInterval(response.expiresIn))
        // Track when this refresh occurred
        lastRefreshTime = Date()
    }

    // MARK: - Clear All

    func clearAll() {
        accessToken = nil
        refreshToken = nil
        userEmail = nil
        userId = nil
        tokenExpirationDate = nil
        lastRefreshTime = nil
    }

    // MARK: - Check if Logged In

    var hasValidSession: Bool {
        return accessToken != nil && userId != nil
    }

    // MARK: - Private Keychain Operations

    private func save(key: String, value: String) {
        guard let data = value.data(using: .utf8) else { return }

        // Delete any existing item first
        delete(key: key)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            // Use AfterFirstUnlockThisDeviceOnly for better accessibility when app resumes
            // This ensures tokens remain accessible after device unlock even if app was killed
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        let status = SecItemAdd(query as CFDictionary, nil)
        if status != errSecSuccess {
            print("Keychain save error for \(key): \(status)")
        }
    }

    private func read(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            return nil
        }

        return value
    }

    private func delete(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        SecItemDelete(query as CFDictionary)
    }
}
