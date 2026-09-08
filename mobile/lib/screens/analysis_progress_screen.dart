import 'package:flutter/material.dart';

import '../models/analysis.dart';
import '../models/crop.dart';
import '../services/analysis_service.dart';
import '../state/app_scope.dart';
import '../state/app_state.dart';
import '../theme/app_theme.dart';
import '../widgets/agri_card.dart';
import 'result_screen.dart';

/// Analysis (loading) screen — previews the three guarded pipeline
/// stages: image quality check, crop verification, disease analysis.
///
/// The stage progression is a UI preview driven by [DemoAnalysisService]
/// timing; real results arrive once the FastAPI backend is connected.
class AnalysisProgressScreen extends StatefulWidget {
  const AnalysisProgressScreen({super.key});

  static const String routeName = '/analysis-progress';

  @override
  State<AnalysisProgressScreen> createState() => _AnalysisProgressScreenState();
}

class _AnalysisProgressScreenState extends State<AnalysisProgressScreen> {
  static const List<AnalysisStage> _stages = AnalysisStage.values;
  static const Duration _stageDelay = Duration(milliseconds: 900);

  AppState? _app;
  bool _started = false;
  int _completedStages = 0;
  bool _finished = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Inherited-widget access belongs here (not in initState); the
    // reference is stored so no context is touched across async gaps.
    _app = AppScope.of(context);
    if (!_started) {
      _started = true;
      _runPreview();
    }
  }

  Future<void> _runPreview() async {
    final app = _app!;
    final request = app.activeRequest ??
        AnalysisRequest(crop: app.selectedCrop, isDemoImage: true);

    final AnalysisService service =
        (request.isDemoImage && (request.imageBytes == null))
            ? const DemoAnalysisService()
            : app.analysisService;

    // Run backend analysis concurrently with stage animations
    final Future<AnalysisResult> analysisFuture = service.analyzeLeaf(request);

    for (int i = 0; i < _stages.length; i++) {
      await Future<void>.delayed(_stageDelay);
      if (!mounted) return;
      setState(() => _completedStages = i + 1);
    }

    final AnalysisResult result = await analysisFuture;
    if (!mounted) return;
    app.setLastResult(result);
    setState(() => _finished = true);
  }

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final app = _app ?? AppScope.of(context);

    return Scaffold(
      appBar: AppBar(title: Text(s.analyzingLeaf)),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                children: [
                  AgriCard(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Text(
                          '${app.selectedCrop.emoji}  ${app.selectedCrop.label}',
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 16),
                        for (int i = 0; i < _stages.length; i++) ...[
                          if (i > 0) const SizedBox(height: 14),
                          _StageRow(
                            label: switch (_stages[i]) {
                              AnalysisStage.qualityCheck => s.qualityCheck,
                              AnalysisStage.cropVerification =>
                                s.cropVerification,
                              AnalysisStage.diseaseAnalysis =>
                                s.diseaseAnalysis,
                            },
                            state: _stageState(i),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  DemoNoticeBanner(message: s.analysisPreviewNote),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: AnimatedOpacity(
                duration: const Duration(milliseconds: 250),
                opacity: _finished ? 1 : 0,
                child: FilledButton(
                  onPressed: _finished
                      ? () => Navigator.of(context)
                          .pushReplacementNamed(ResultScreen.routeName)
                      : null,
                  child: Text(s.viewResult),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  _StageState _stageState(int index) {
    if (index < _completedStages) return _StageState.done;
    if (index == _completedStages && !_finished) return _StageState.active;
    return _StageState.pending;
  }
}

enum _StageState { pending, active, done }

class _StageRow extends StatelessWidget {
  const _StageRow({required this.label, required this.state});

  final String label;
  final _StageState state;

  @override
  Widget build(BuildContext context) {
    final Color color = switch (state) {
      _StageState.done => AgriColors.primary,
      _StageState.active => Colors.green.shade700,
      _StageState.pending => Colors.black26,
    };

    return Row(
      children: [
        _stageIcon(color),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            label,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  color: state == _StageState.pending
                      ? Colors.black45
                      : Colors.black87,
                  fontWeight: state == _StageState.pending
                      ? FontWeight.w400
                      : FontWeight.w600,
                ),
          ),
        ),
      ],
    );
  }

  Widget _stageIcon(Color color) {
    switch (state) {
      case _StageState.done:
        return Icon(Icons.check_circle, color: color, size: 26);
      case _StageState.active:
        return SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(strokeWidth: 2.4, color: color),
        );
      case _StageState.pending:
        return Icon(Icons.radio_button_unchecked, color: color, size: 24);
    }
  }
}
