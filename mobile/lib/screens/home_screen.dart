import 'package:flutter/material.dart';

import '../models/language.dart';
import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import '../widgets/agri_card.dart';
import '../widgets/brand.dart';
import 'assistant_screen.dart';
import 'crop_selection_screen.dart';
import 'language_screen.dart';

/// Home screen — branding, short explanation, primary and secondary
/// actions, and the About section.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  static const String routeName = '/home';

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const BrandMark(size: 34, iconSize: 18),
            const SizedBox(width: 10),
            Text(s.appName),
          ],
        ),
        actions: [
          IconButton(
            tooltip: s.language,
            icon: const Icon(Icons.translate),
            onPressed: () =>
                Navigator.of(context).pushNamed(LanguageScreen.routeName),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            // -- Brand / headline card --------------------------------
            AgriCard(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    s.homeHeadline,
                    style: textTheme.headlineSmall?.copyWith(
                      color: AgriColors.heading,
                      fontWeight: FontWeight.w800,
                      height: 1.2,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    s.homeDescription,
                    style: textTheme.bodyMedium?.copyWith(
                      color: Colors.black87,
                      height: 1.45,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -- Primary + secondary actions --------------------------
            FilledButton.icon(
              icon: const Icon(Icons.search),
              label: Text(s.analyzeCrop),
              onPressed: () => Navigator.of(context)
                  .pushNamed(CropSelectionScreen.routeName),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              icon: const Icon(Icons.forum_outlined),
              label: Text(s.askAgriMind),
              onPressed: () =>
                  Navigator.of(context).pushNamed(AssistantScreen.routeName),
            ),
            const SizedBox(height: 24),

            // -- How it works ------------------------------------------
            AgriCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SectionHeader(s.howItWorks),
                  const SizedBox(height: 14),
                  WorkflowStepRow(
                    number: 1,
                    title: s.selectYourCrop,
                    subtitle: '${s.cropLabel}: 🍅 · 🥔 · 🍎',
                  ),
                  const SizedBox(height: 12),
                  WorkflowStepRow(
                    number: 2,
                    title: s.uploadLeafPhoto,
                    subtitle: s.takePhotoHelp,
                  ),
                  const SizedBox(height: 12),
                  WorkflowStepRow(
                    number: 3,
                    title: s.qualityCheck,
                    subtitle: s.cropVerification,
                  ),
                  const SizedBox(height: 12),
                  WorkflowStepRow(
                    number: 4,
                    title: s.diagnosisTitle,
                    subtitle: s.gradcamTitle,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -- Language quick link -----------------------------------
            AgriCard(
              onTap: () =>
                  Navigator.of(context).pushNamed(LanguageScreen.routeName),
              child: Row(
                children: [
                  const Icon(Icons.translate, color: AgriColors.primary),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      '${s.appLanguage}: ${AppScope.of(context).language.nativeLabel}',
                      style: textTheme.titleSmall
                          ?.copyWith(fontWeight: FontWeight.w600),
                    ),
                  ),
                  const Icon(Icons.chevron_right, color: Colors.black38),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -- About --------------------------------------------------
            AgriCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SectionHeader(s.aboutTitle),
                  const SizedBox(height: 10),
                  Text(
                    s.aboutBody,
                    style: textTheme.bodySmall?.copyWith(
                      color: Colors.black87,
                      height: 1.5,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Alibaba Cloud AI Hackathon Pakistan 2026 · Mobile MVP',
                    style: textTheme.bodySmall?.copyWith(
                      color: Colors.black45,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
