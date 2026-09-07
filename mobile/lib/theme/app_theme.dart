import 'package:flutter/material.dart';

/// AgriMind AI visual identity.
///
/// Green agricultural branding carried over from the existing Streamlit
/// web app (primary #2E7D32, light surface #F1F8E9).  The theme stays
/// close to Material 3 defaults: calm, readable, professional — no
/// excessive gradients or flashy dashboard styling.
abstract final class AgriColors {
  /// Primary brand green (green 800).
  static const Color primary = Color(0xFF2E7D32);

  /// Deep green used for headings on light surfaces (green 900).
  static const Color heading = Color(0xFF1B5E20);

  /// Light green scaffold background (matches the web app).
  static const Color scaffoldBackground = Color(0xFFF1F8E9);

  /// Soft green card tint for emphasis surfaces (green 50).
  static const Color softGreen = Color(0xFFE8F5E9);
}

/// Application theme for AgriMind AI.
abstract final class AgriTheme {
  static ThemeData get light {
    final ColorScheme scheme = ColorScheme.fromSeed(
      seedColor: AgriColors.primary,
      primary: AgriColors.primary,
      surface: Colors.white,
      surfaceContainerLowest: Colors.white,
    );

    final ThemeData base = ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: AgriColors.scaffoldBackground,
      appBarTheme: const AppBarTheme(
        backgroundColor: AgriColors.scaffoldBackground,
        foregroundColor: AgriColors.heading,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: AgriColors.heading,
          fontSize: 20,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
        ),
      ),
    );

    return base.copyWith(
      cardTheme: CardThemeData(
        color: Colors.white,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: Colors.green.withValues(alpha: 0.10)),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: AgriColors.primary,
          foregroundColor: Colors.white,
          minimumSize: const Size.fromHeight(52),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AgriColors.primary,
          minimumSize: const Size.fromHeight(52),
          side: const BorderSide(color: AgriColors.primary, width: 1.2),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.green.shade200),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.green.shade200),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AgriColors.primary, width: 1.6),
        ),
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: AgriColors.heading,
        contentTextStyle: const TextStyle(color: Colors.white, fontSize: 14),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
      dividerTheme: DividerThemeData(
        color: Colors.green.shade100,
        thickness: 1,
        space: 1,
      ),
    );
  }
}
