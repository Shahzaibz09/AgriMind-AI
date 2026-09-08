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
    this.gradcamOverlayBase64,
    this.probabilities,
    this.cautions = const <String>[],
    this.statusLabel,
    this.description,
    this.apiStatus,
    this.errorMessage,
    this.detectedCrop,
  });

  final Crop crop;
  final String? disease;
  final double? confidence;
  final AnalysisStatus status;
  final String? recommendedNextStep;

  /// Base64 encoded Grad-CAM overlay image (data:image/jpeg;base64,...).
  final String? gradcamOverlayBase64;

  /// Softmax probability dictionary (Class -> Probability).
  final Map<String, double>? probabilities;

  /// Uncertainty or quality caution notes.
  final List<String> cautions;

  /// Result status label (e.g. "Healthy", "Action Recommended").
  final String? statusLabel;

  /// Detailed description from knowledge base.
  final String? description;

  /// Status code from API ("success", "quality_rejected", "crop_mismatch", "error").
  final String? apiStatus;

  /// Human-readable error message if analysis could not complete.
  final String? errorMessage;

  /// Detected crop name if mismatch occurred.
  final String? detectedCrop;

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

  /// Parses JSON response from the AgriMind AI FastAPI backend.
  factory AnalysisResult.fromJson(Map<String, dynamic> json, Crop selectedCrop) {
    final statusStr = json['status'] as String? ?? 'error';
    if (statusStr == 'success') {
      final probs = (json['probabilities'] as Map<String, dynamic>?)?.map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      );
      final cautionsList = (json['cautions'] as List<dynamic>?)
              ?.map((e) => e.toString())
              .toList() ??
          const <String>[];
      return AnalysisResult(
        crop: selectedCrop,
        isPlaceholder: false,
        disease: json['predicted_disease'] as String?,
        confidence: (json['confidence'] as num?)?.toDouble(),
        status: AnalysisStatus.reviewed,
        statusLabel: json['status_label'] as String?,
        description: json['description'] as String?,
        recommendedNextStep: json['recommended_next_step'] as String?,
        gradcamOverlayBase64: json['gradcam_overlay_base64'] as String?,
        probabilities: probs,
        cautions: cautionsList,
        apiStatus: 'success',
      );
    } else if (statusStr == 'quality_rejected') {
      return AnalysisResult(
        crop: selectedCrop,
        isPlaceholder: false,
        status: AnalysisStatus.reviewed,
        statusLabel: 'Quality Check Failed',
        apiStatus: 'quality_rejected',
        errorMessage: json['message'] as String? ?? 'Image quality check failed.',
        recommendedNextStep: json['recommended_next_step'] as String? ??
            'Please take a clear, well-lit close-up of a leaf.',
      );
    } else if (statusStr == 'crop_mismatch' || statusStr == 'crop_uncertain') {
      return AnalysisResult(
        crop: selectedCrop,
        isPlaceholder: false,
        status: AnalysisStatus.reviewed,
        statusLabel: 'Crop Verification Alert',
        apiStatus: statusStr,
        detectedCrop: json['detected_crop'] as String?,
        errorMessage: json['message'] as String? ?? 'Crop verification failed.',
        recommendedNextStep: json['recommended_next_step'] as String?,
      );
    } else {
      return AnalysisResult(
        crop: selectedCrop,
        isPlaceholder: false,
        status: AnalysisStatus.reviewed,
        statusLabel: 'Error',
        apiStatus: 'error',
        errorMessage: json['message'] as String? ?? 'Analysis could not be completed.',
      );
    }
  }

  /// Connection or parsing error result.
  factory AnalysisResult.error(Crop crop, String message) {
    return AnalysisResult(
      crop: crop,
      isPlaceholder: false,
      status: AnalysisStatus.reviewed,
      statusLabel: 'Connection Error',
      apiStatus: 'error',
      errorMessage: message,
      recommendedNextStep:
          'Ensure the AgriMind AI backend server is running and reachable.',
    );
  }
}
