import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Circular brand mark used on the splash screen and headers.
class BrandMark extends StatelessWidget {
  const BrandMark({super.key, this.size = 88, this.iconSize = 44});

  final double size;
  final double iconSize;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: AgriColors.primary,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: AgriColors.primary.withValues(alpha: 0.25),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Icon(Icons.eco, color: Colors.white, size: iconSize),
    );
  }
}

/// One numbered "How it works" step row.
class WorkflowStepRow extends StatelessWidget {
  const WorkflowStepRow({
    super.key,
    required this.number,
    required this.title,
    required this.subtitle,
  });

  final int number;
  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 28,
          height: 28,
          alignment: Alignment.center,
          decoration: const BoxDecoration(
            color: AgriColors.softGreen,
            shape: BoxShape.circle,
          ),
          child: Text(
            '$number',
            style: const TextStyle(
              color: AgriColors.heading,
              fontWeight: FontWeight.w700,
              fontSize: 13,
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context)
                    .textTheme
                    .titleSmall
                    ?.copyWith(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 2),
              Text(
                subtitle,
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: Colors.black54),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
