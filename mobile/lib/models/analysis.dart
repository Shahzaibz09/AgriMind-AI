import 'dart:typed_data';

import 'crop.dart';

/// Input for one leaf-analysis run.
///
/// In Step 1 the UI only supplies demo inputs; once the FastAPI backend
/// exists, [imageBytes] will carry the captured or gallery-picked photo.
class AnalysisRequest {
  const AnalysisRequest({
    required this.crop,
    this.imageBytes,
    this.imageName,
    this.isDemoImage = false,
  });

  final Crop crop;
  final Uint8List? imageBytes;
  final String? imageName;
  final bool isDemoImage;
}

/// Stages of the guarded analysis pipeline, in execution order.
///
/// Mirrors the workflow communicated by the Streamlit app: quality gate →
/// crop verification → disease prediction.
enum AnalysisStage {
  qualityCheck,
  cropVerification,
  diseaseAnalysis,
}

/// Overall outcome status shown on the result card.
enum AnalysisStatus { pending, reviewed }

/// One completed leaf analysis.
///
/// Step 1 never fabricates disease data: the [AnalysisService] produces
/// instances with [isPlaceholder] `true` and empty diagnostic fields,
/// which the result screen renders as clearly marked placeholders.
class AnalysisResult {
  const AnalysisResult({
    required this.crop,
    required this.isPlaceholder,
    this.disease,
    this.confidence,
    this.status = AnalysisStatus.pending,
    this.recommendedNextStep,
  });

  final Crop crop;
  final String? disease;
  final double? confidence;
  final AnalysisStatus status;
  final String? recommendedNextStep;

  /// `true` while no real backend data is available.
  final bool isPlaceholder;

  /// Placeholder result used by the demo flow (Step 1).
  factory AnalysisResult.placeholderFor(Crop crop) {
    return AnalysisResult(
      crop: crop,
      isPlaceholder: true,
      disease: null,
      confidence: null,
      status: AnalysisStatus.pending,
      recommendedNextStep: null,
    );
  }
}
