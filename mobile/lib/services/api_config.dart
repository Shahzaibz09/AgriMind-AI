import 'dart:io' show Platform;
import 'package:flutter/foundation.dart';

/// Configuration for connecting to the AgriMind AI FastAPI backend.
class ApiConfig {
  ApiConfig._();

  /// Standard loopback endpoints.
  static const String emulatorBaseUrl = 'http://10.0.2.2:8000';
  static const String physicalDeviceBaseUrl = 'http://127.0.0.1:8000';
  static const String defaultLocalUrl = 'http://localhost:8000';

  /// Compile-time build overrides via --dart-define.
  static const String _envBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: String.fromEnvironment('AGRIMIND_API_URL'),
  );
  static const bool _usePhysicalDevice =
      bool.fromEnvironment('PHYSICAL_DEVICE', defaultValue: false) ||
      bool.fromEnvironment('USE_ADB_REVERSE', defaultValue: false);

  static String _baseUrl = _resolveDefaultUrl();

  static String _resolveDefaultUrl() {
    if (_envBaseUrl.isNotEmpty) {
      return _sanitize(_envBaseUrl);
    }
    if (kIsWeb) {
      return defaultLocalUrl;
    }
    try {
      if (Platform.isAndroid) {
        if (_usePhysicalDevice) {
          // Physical Android device connected via ADB reverse tunnel
          return physicalDeviceBaseUrl;
        }
        // Standard Android emulator loopback to host machine
        return emulatorBaseUrl;
      }
    } catch (_) {
      // Platform check may fail on unsupported targets
    }
    return defaultLocalUrl;
  }

  static String _sanitize(String url) =>
      url.endsWith('/') ? url.substring(0, url.length - 1) : url;

  /// Current base URL of the FastAPI backend.
  static String get baseUrl => _baseUrl;

  /// Override the backend URL (e.g. for physical devices on a local Wi-Fi network).
  static set baseUrl(String url) {
    _baseUrl = _sanitize(url);
  }

  /// Switch active backend target to physical Android device (ADB reverse).
  static void usePhysicalDevice() => _baseUrl = physicalDeviceBaseUrl;

  /// Switch active backend target to Android emulator.
  static void useEmulator() => _baseUrl = emulatorBaseUrl;

  /// Reset active base URL to the default resolved environment value.
  static void resetToDefault() => _baseUrl = _resolveDefaultUrl();
}
