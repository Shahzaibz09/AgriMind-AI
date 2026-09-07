import 'package:flutter/material.dart';

import '../models/language.dart';
import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import '../widgets/agri_card.dart';

/// Language selection screen — English, اردو and Roman Urdu.
///
/// Selecting a language updates [AppState] immediately; the whole app
/// (including text direction for Urdu) rebuilds through [AppScope].
class LanguageScreen extends StatelessWidget {
  const LanguageScreen({super.key});

  static const String routeName = '/language';

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final app = AppScope.of(context);

    return Scaffold(
      appBar: AppBar(title: Text(s.appLanguage)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            for (final AppLanguage language in AppLanguage.values)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: _LanguageTile(
                  language: language,
                  selected: language == app.language,
                  onTap: () {
                    app.setLanguage(language);
                    ScaffoldMessenger.of(context)
                      ..hideCurrentSnackBar()
                      ..showSnackBar(
                        SnackBar(content: Text(s.languageChanged)),
                      );
                  },
                ),
              ),
            const SizedBox(height: 8),
            AgriCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SectionHeader(
                    s.appLanguage,
                    subtitle: '${app.language.nativeLabel} · '
                        '${app.language.code}',
                  ),
                  const SizedBox(height: 10),
                  Text(
                    s.homeHeadline,
                    style: Theme.of(context)
                        .textTheme
                        .bodyMedium
                        ?.copyWith(color: AgriColors.heading),
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

class _LanguageTile extends StatelessWidget {
  const _LanguageTile({
    required this.language,
    required this.selected,
    required this.onTap,
  });

  final AppLanguage language;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return AgriCard(
      onTap: onTap,
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          // Selection indicator (tapping anywhere on the tile selects).
          Icon(
            selected ? Icons.check_circle : Icons.radio_button_unchecked,
            color: selected ? AgriColors.primary : Colors.black26,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  language.label,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                        color:
                            selected ? AgriColors.heading : Colors.black87,
                      ),
                ),
                const SizedBox(height: 2),
                Text(
                  language.description,
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: Colors.black54),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
