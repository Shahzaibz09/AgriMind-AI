import 'package:flutter/material.dart';

import '../models/crop.dart';
import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import '../widgets/agri_card.dart';
import 'leaf_analysis_screen.dart';

/// Crop selection screen — Tomato, Potato and Apple tiles with the
/// current selection clearly highlighted.
class CropSelectionScreen extends StatelessWidget {
  const CropSelectionScreen({super.key});

  static const String routeName = '/crops';

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final app = AppScope.of(context);

    return Scaffold(
      appBar: AppBar(title: Text(s.selectYourCrop)),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                children: [
                  for (final Crop crop in Crop.values)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: _CropTile(
                        crop: crop,
                        selected: crop == app.selectedCrop,
                        onTap: () => app.selectCrop(crop),
                      ),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: FilledButton(
                onPressed: () => Navigator.of(context)
                    .pushNamed(LeafAnalysisScreen.routeName),
                child: Text(s.continueLabel),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CropTile extends StatelessWidget {
  const _CropTile({
    required this.crop,
    required this.selected,
    required this.onTap,
  });

  final Crop crop;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final textTheme = Theme.of(context).textTheme;

    return AgriCard(
      onTap: onTap,
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          Text(crop.emoji, style: const TextStyle(fontSize: 34)),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  crop.label,
                  style: textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: selected ? AgriColors.heading : Colors.black87,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  crop.tagline,
                  style: textTheme.bodySmall?.copyWith(color: Colors.black54),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: selected ? AgriColors.primary : Colors.transparent,
              borderRadius: BorderRadius.circular(999),
              border: Border.all(
                color: selected ? AgriColors.primary : Colors.black26,
              ),
            ),
            child: Text(
              s.selected,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: selected ? Colors.white : Colors.black54,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
