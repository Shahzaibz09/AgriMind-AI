import 'dart:io' show Platform;
import 'package:flutter/foundation.dart';

/// Configuration for connecting to the AgriMind AI FastAPI backend.
class ApiConfig {
  ApiConfig._();

  static String _baseUrl = _resolveDefaultUrl();

  static String _resolveDefaultUrl() {
    if (kIsWeb) {
      return 'http://localhost:8000';
    }
    try {
      if (Platform.isAndroid) {
        // Standard Android emulator loopback to host machine
        return 'http://10.0.2.2:8000';
      }
    } catch (_) {
      // Platform check may fail on unsupported targets
    }
    return 'http://localhost:8000';
  }

  /// Current base URL of the FastAPI backend.
  static String get baseUrl => _baseUrl;

  /// Override the backend URL (e.g. for physical devices on a local Wi-Fi network).
  static set baseUrl(String url) {
    _baseUrl = url.endsWith('/') ? url.substring(0, url.length - 1) : url;
  }
}
