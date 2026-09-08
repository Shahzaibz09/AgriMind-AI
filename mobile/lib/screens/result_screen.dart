import 'dart:convert';
import 'package:flutter/material.dart';

import '../models/analysis.dart';
import '../models/crop.dart';
import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import '../widgets/agri_card.dart';
import 'leaf_analysis_screen.dart';

/// Result screen — diagnosis card structure (crop, disease, confidence,
/// status, Grad-CAM area, recommended next step) rendered from the
/// latest [AnalysisResult].
class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  static const String routeName = '/result';

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final app = AppScope.of(context);
    final AnalysisResult? result = app.lastResult;
    final textTheme = Theme.of(context).textTheme;

    final String cropLabel = result?.crop.label ?? app.selectedCrop.label;
    final String cropEmoji = result?.crop.emoji ?? app.selectedCrop.emoji;
    final bool isPlaceholder = result?.isPlaceholder ?? true;

    // Decode base64 Grad-CAM overlay if available
    Widget? gradcamWidget;
    if (!isPlaceholder && result?.gradcamOverlayBase64 != null) {
      try {
        final b64String = result!.gradcamOverlayBase64!.contains(',')
            ? result.gradcamOverlayBase64!.split(',').last
            : result.gradcamOverlayBase64!;
        final bytes = base64Decode(b64String);
        gradcamWidget = ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.memory(
            bytes,
            height: 220,
            width: double.infinity,
            fit: BoxFit.contain,
          ),
        );
      } catch (_) {
        gradcamWidget = null;
      }
    }

    return Scaffold(
      appBar: AppBar(title: Text(s.diagnosisTitle)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            // -- Placeholder banner (only when in demo mode) ------------
            if (isPlaceholder) ...[
              DemoNoticeBanner(message: s.placeholderNotice),
              const SizedBox(height: 16),
            ],

            // -- Alert banner (if quality rejected or crop mismatch) ----
            if (!isPlaceholder &&
                result?.apiStatus != null &&
                result!.apiStatus != 'success') ...[
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.orange.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.orange.shade300),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.warning_amber_rounded,
                        color: Colors.orange.shade800),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            result.statusLabel ?? 'Alert',
                            style: textTheme.titleSmall?.copyWith(
                              fontWeight: FontWeight.w700,
                              color: Colors.orange.shade900,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            result.errorMessage ?? '',
                            style: textTheme.bodySmall?.copyWith(
                              color: Colors.black87,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],

            // -- Diagnosis card -----------------------------------------
            AgriCard(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(cropEmoji, style: const TextStyle(fontSize: 30)),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          '$cropLabel — ${s.diagnosisTitle}',
                          style: textTheme.titleLarge?.copyWith(
                            color: AgriColors.heading,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const Divider(height: 28),
                  _ResultRow(
                    label: s.cropLabel,
                    value: cropLabel,
                  ),
                  _ResultRow(
                    label: s.diseaseLabel,
                    value: isPlaceholder
                        ? s.notAvailableYet
                        : (result?.disease ?? s.notAvailableYet),
                    muted: isPlaceholder,
                  ),
                  _ResultRow(
                    label: s.confidenceLabel,
                    value: isPlaceholder
                        ? s.notAvailableYet
                        : (result?.confidence != null
                            ? '${(result!.confidence! * 100).toStringAsFixed(1)}%'
                            : s.notAvailableYet),
                    muted: isPlaceholder,
                  ),
                  _ResultRow(
                    label: s.statusLabel,
                    value: isPlaceholder
                        ? s.notAvailableYet
                        : (result?.statusLabel ?? s.notAvailableYet),
                    muted: isPlaceholder,
                  ),
                  _ResultRow(
                    label: s.recommendedNextStep,
                    value: isPlaceholder
                        ? s.notAvailableYet
                        : (result?.recommendedNextStep ?? s.notAvailableYet),
                    muted: isPlaceholder,
                    isLast: true,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -- Grad-CAM area -------------------------------------------
            AgriCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SectionHeader(s.gradcamTitle),
                  const SizedBox(height: 12),
                  gradcamWidget ??
                      Container(
                        height: 160,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          color: Colors.green.shade50,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.green.shade200),
                        ),
                        child: Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.gradient,
                                  size: 34, color: Colors.green.shade300),
                              const SizedBox(height: 6),
                              Text(
                                s.notAvailableYet,
                                style: textTheme.bodySmall
                                    ?.copyWith(color: Colors.black45),
                              ),
                            ],
                          ),
                        ),
                      ),
                  const SizedBox(height: 10),
                  _GradcamLegend(
                    lessLabel: s.lessInfluence,
                    moreLabel: s.moreInfluence,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -- Uncertainty Cautions (if any) ---------------------------
            if (!isPlaceholder &&
                result?.cautions != null &&
                result!.cautions.isNotEmpty) ...[
              AgriCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.info_outline,
                            size: 18, color: Colors.orange.shade700),
                        const SizedBox(width: 8),
                        Text(
                          'Prediction Notes',
                          style: textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w700,
                            color: Colors.orange.shade900,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    for (final caution in result.cautions)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text(
                          '• $caution',
                          style: textTheme.bodySmall?.copyWith(
                            color: Colors.black87,
                            height: 1.4,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
            ],

            // -- AI disclaimer -------------------------------------------
            AgriCard(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.shield_outlined,
                      size: 20, color: Colors.green.shade700),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      s.aboutBody,
                      style: textTheme.bodySmall?.copyWith(height: 1.5),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // -- Reset action --------------------------------------------
            FilledButton.icon(
              icon: const Icon(Icons.refresh),
              label: Text(s.analyzeAnotherLeaf),
              onPressed: () {
                // Clears the analysis only — crop and language stay.
                app.resetAnalysis();
                Navigator.of(context).pushNamedAndRemoveUntil(
                  LeafAnalysisScreen.routeName,
                  (Route<dynamic> route) => route.isFirst,
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultRow extends StatelessWidget {
  const _ResultRow({
    required this.label,
    required this.value,
    this.muted = false,
    this.isLast = false,
  });

  final String label;
  final String value;
  final bool muted;
  final bool isLast;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 150,
              child: Text(
                label,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.black54,
                      fontWeight: FontWeight.w600,
                    ),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: muted ? Colors.black45 : Colors.black87,
                      fontWeight: muted ? FontWeight.w400 : FontWeight.w700,
                    ),
              ),
            ),
          ],
        ),
        if (!isLast) const SizedBox(height: 14),
      ],
    );
  }
}

/// Blue → green → yellow → red influence legend (presentation only).
class _GradcamLegend extends StatelessWidget {
  const _GradcamLegend({required this.lessLabel, required this.moreLabel});

  final String lessLabel;
  final String moreLabel;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 10,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(5),
            gradient: LinearGradient(
              colors: [
                Colors.blue.shade300,
                Colors.cyan.shade300,
                Colors.green.shade300,
                Colors.yellow.shade600,
                Colors.red.shade400,
              ],
            ),
          ),
        ),
        const SizedBox(height: 6),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              lessLabel,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Colors.black54),
            ),
            Text(
              moreLabel,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Colors.black54),
            ),
          ],
        ),
      ],
    );
  }
}
