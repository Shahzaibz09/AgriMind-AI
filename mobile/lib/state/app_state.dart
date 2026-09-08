import 'package:flutter/foundation.dart';

import '../l10n/app_strings.dart';
import '../models/analysis.dart';
import '../models/crop.dart';
import '../models/language.dart';
import '../services/analysis_service.dart';
import '../services/assistant_service.dart';
import '../services/http_analysis_service.dart';
import '../services/http_assistant_service.dart';

/// App-wide session state: selected crop, selected language, active
/// request, and the latest analysis result.
class AppState extends ChangeNotifier {
  AppState({
    AnalysisService? analysisService,
    AssistantService? assistantService,
  })  : _analysisService = analysisService ?? HttpAnalysisService(),
        _assistantService = assistantService ?? HttpAssistantService();

  final AnalysisService _analysisService;
  final AssistantService _assistantService;

  AnalysisService get analysisService => _analysisService;
  AssistantService get assistantService => _assistantService;

  Crop _selectedCrop = Crop.tomato;
  AppLanguage _language = AppLanguage.english;
  AnalysisResult? _lastResult;
  AnalysisRequest? _activeRequest;

  /// Currently selected crop (Tomato default, like the web app).
  Crop get selectedCrop => _selectedCrop;

  /// Currently selected UI/assistant language.
  AppLanguage get language => _language;

  /// Latest analysis result, if an analysis was completed.
  AnalysisResult? get lastResult => _lastResult;

  /// Active request pending or currently analyzing.
  AnalysisRequest? get activeRequest => _activeRequest;

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

  void setActiveRequest(AnalysisRequest? request) {
    _activeRequest = request;
    notifyListeners();
  }

  /// Stores a completed analysis.
  void setLastResult(AnalysisResult result) {
    _lastResult = result;
    notifyListeners();
  }

  /// Clears analysis state only — crop and language selections are kept,
  /// mirroring the web app's "Analyze Another Leaf" reset.
  void resetAnalysis() {
    _lastResult = null;
    _activeRequest = null;
    notifyListeners();
  }
}
