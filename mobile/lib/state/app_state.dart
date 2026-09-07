import 'package:flutter/foundation.dart';

import '../l10n/app_strings.dart';
import '../models/analysis.dart';
import '../models/crop.dart';
import '../models/language.dart';

/// App-wide session state: selected crop, selected language, and the
/// latest (placeholder) analysis result.
///
/// Keeping this outside the widgets lets any screen read and update the
/// selection without prop-drilling, and makes the future backend
/// integration a matter of replacing service implementations.
class AppState extends ChangeNotifier {
  Crop _selectedCrop = Crop.tomato;
  AppLanguage _language = AppLanguage.english;
  AnalysisResult? _lastResult;

  /// Currently selected crop (Tomato default, like the web app).
  Crop get selectedCrop => _selectedCrop;

  /// Currently selected UI/assistant language.
  AppLanguage get language => _language;

  /// Latest analysis result, if a (demo) analysis was completed.
  AnalysisResult? get lastResult => _lastResult;

  /// Localized UI strings for the current language.
  AppStrings get strings => AppStrings.forLanguage(_language);

  void selectCrop(Crop crop) {
    if (crop == _selectedCrop) return;
    _selectedCrop = crop;
    notifyListeners();
  }

  void setLanguage(AppLanguage language) {
    if (language == _language) return;
    _language = language;
    notifyListeners();
  }

  /// Stores a completed (placeholder) analysis.
  void setLastResult(AnalysisResult result) {
    _lastResult = result;
    notifyListeners();
  }

  /// Clears analysis state only — crop and language selections are kept,
  /// mirroring the web app's "Analyze Another Leaf" reset.
  void resetAnalysis() {
    _lastResult = null;
    notifyListeners();
  }
}
