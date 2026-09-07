import '../models/analysis.dart';

/// Leaf-analysis contract for the AgriMind AI mobile app.
///
/// Step 2 will provide an implementation that talks to the planned
/// FastAPI backend, which itself reuses the existing Python/PyTorch
/// inference pipeline:
///
/// ```text
/// Flutter  →  FastAPI  →  existing src/inference.py + Grad-CAM
/// ```
///
/// Expected future endpoints (to be implemented in Step 2 — NOT now):
///  - `POST /analyze`   — multipart image + crop; returns quality,
///                        crop verification, diagnosis, Grad-CAM image.
///
/// The UI depends only on this interface, so swapping the demo
/// implementation for the real backend requires no screen changes.
abstract interface class AnalysisService {
  /// Runs the guarded analysis pipeline for [request].
  ///
  /// Implementations must never fabricate diagnoses: when no real
  /// backend is available, return a result with `isPlaceholder == true`.
  Future<AnalysisResult> analyzeLeaf(AnalysisRequest request);
}

/// Step 1 placeholder implementation.
///
/// Simulates the pipeline delay and returns a clearly marked
/// placeholder result — no disease names, confidence values, or other
/// AI output are invented.
class DemoAnalysisService implements AnalysisService {
  const DemoAnalysisService({this.simulatedDelay = const Duration(milliseconds: 900)});

  /// Per-stage delay used to preview the analysis flow in the UI.
  final Duration simulatedDelay;

  @override
  Future<AnalysisResult> analyzeLeaf(AnalysisRequest request) async {
    await Future<void>.delayed(simulatedDelay);
    return AnalysisResult.placeholderFor(request.crop);
  }
}
