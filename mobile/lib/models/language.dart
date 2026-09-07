import 'package:flutter/widgets.dart';

/// Assistant/UI languages supported by AgriMind AI.
///
/// Mirrors the language options of the existing Streamlit web app:
/// English, Urdu (اردو) and Roman Urdu.  This enum only establishes the
/// selection architecture — knowledge content is not duplicated here.
enum AppLanguage { english, urdu, romanUrdu }

extension AppLanguageInfo on AppLanguage {
  /// Stable code, shared with the web app's language codes.
  String get code {
    return switch (this) {
      AppLanguage.english => 'en',
      AppLanguage.urdu => 'ur',
      AppLanguage.romanUrdu => 'roman_ur',
    };
  }

  /// Label shown in language pickers.
  String get label {
    return switch (this) {
      AppLanguage.english => 'English',
      AppLanguage.urdu => 'اردو (Urdu)',
      AppLanguage.romanUrdu => 'Roman Urdu',
    };
  }

  /// Short native name for compact chips.
  String get nativeLabel {
    return switch (this) {
      AppLanguage.english => 'English',
      AppLanguage.urdu => 'اردو',
      AppLanguage.romanUrdu => 'Roman Urdu',
    };
  }

  /// Helper description shown in the language screen.
  String get description {
    return switch (this) {
      AppLanguage.english => 'Guidance and interface in English',
      AppLanguage.urdu => 'رہنمائی اور انٹرفیس اردو میں',
      AppLanguage.romanUrdu =>
        'Rehnumai aur interface Roman Urdu mein (English letters)',
    };
  }

  /// Urdu is written right-to-left.
  TextDirection get textDirection {
    return switch (this) {
      AppLanguage.urdu => TextDirection.rtl,
      _ => TextDirection.ltr,
    };
  }
}
