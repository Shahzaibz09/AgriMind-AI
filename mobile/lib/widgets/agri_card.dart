import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Consistent white card used across AgriMind AI screens.
class AgriCard extends StatelessWidget {
  const AgriCard({super.key, required this.child, this.padding, this.onTap});

  final Widget child;
  final EdgeInsetsGeometry? padding;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final CardThemeData cardTheme = CardTheme.of(context);
    final Widget content = Padding(
      padding: padding ?? const EdgeInsets.all(16),
      child: child,
    );

    if (onTap == null) {
      return Card(child: content);
    }

    return Card(
      clipBehavior: Clip.antiAlias,
      shape: cardTheme.shape,
      child: InkWell(onTap: onTap, child: content),
    );
  }
}

/// Section heading with a green accent bar.
class SectionHeader extends StatelessWidget {
  const SectionHeader(this.title, {super.key, this.subtitle});

  final String title;
  final String? subtitle;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 4,
              height: 20,
              decoration: BoxDecoration(
                color: AgriColors.primary,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                title,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: AgriColors.heading,
                      fontWeight: FontWeight.w700,
                    ),
              ),
            ),
          ],
        ),
        if (subtitle != null) ...[
          const SizedBox(height: 4),
          Padding(
            padding: const EdgeInsets.only(left: 14),
            child: Text(
              subtitle!,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.black54,
                  ),
            ),
          ),
        ],
      ],
    );
  }
}

/// Amber banner marking placeholder / demo-only content.
class DemoNoticeBanner extends StatelessWidget {
  const DemoNoticeBanner({super.key, required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.amber.shade200),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, size: 20, color: Colors.amber.shade800),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.brown.shade800,
                    fontWeight: FontWeight.w500,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}
