import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/analysis.dart';
import '../models/crop.dart';
import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import '../widgets/agri_card.dart';
import 'analysis_progress_screen.dart';

/// Leaf analysis screen for the selected crop.
///
/// Supports capturing photos via Camera, picking from Gallery, or
/// running the Demo Image flow.
class LeafAnalysisScreen extends StatelessWidget {
  const LeafAnalysisScreen({super.key});

  static const String routeName = '/analysis';

  Future<void> _pickImage(BuildContext context, ImageSource source) async {
    final s = AppScope.of(context).strings;
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(
        source: source,
        maxWidth: 1600,
        maxHeight: 1600,
        imageQuality: 90,
      );
      if (picked == null) return;
      final bytes = await picked.readAsBytes();
      if (!context.mounted) return;

      final app = AppScope.of(context);
      app.setActiveRequest(AnalysisRequest(
        crop: app.selectedCrop,
        imageBytes: bytes,
        imageName: picked.name,
        isDemoImage: false,
      ));
      Navigator.of(context).pushNamed(AnalysisProgressScreen.routeName);
    } catch (_) {
      // In automated widget tests or when device camera is unsupported,
      // fallback to the explanatory guidance sheet
      if (context.mounted) {
        _showPlaceholderSheet(
          context,
          source == ImageSource.camera ? s.takePhoto : s.chooseFromGallery,
          s.takePhotoHelp,
          source == ImageSource.camera
              ? Icons.photo_camera_outlined
              : Icons.photo_library_outlined,
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    final app = AppScope.of(context);
    final Crop crop = app.selectedCrop;
    final textTheme = Theme.of(context).textTheme;

    return Scaffold(
      appBar: AppBar(title: Text(s.uploadLeafPhoto)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            // -- Selected crop summary ---------------------------------
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.green.shade200),
              ),
              child: Row(
                children: [
                  Text(crop.emoji, style: const TextStyle(fontSize: 26)),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      '${s.selected}: ${crop.label}',
                      style: textTheme.titleSmall
                          ?.copyWith(fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -- Entry actions ------------------------------------------
            _ActionTile(
              icon: Icons.photo_camera_outlined,
              title: s.takePhoto,
              subtitle: s.takePhotoHelp,
              onTap: () => _showCaptureSheet(
                context,
                s.takePhoto,
                s.takePhotoHelp,
                Icons.photo_camera_outlined,
                ImageSource.camera,
              ),
            ),
            const SizedBox(height: 12),
            _ActionTile(
              icon: Icons.photo_library_outlined,
              title: s.chooseFromGallery,
              subtitle: s.takePhotoHelp,
              onTap: () => _showCaptureSheet(
                context,
                s.chooseFromGallery,
                s.takePhotoHelp,
                Icons.photo_library_outlined,
                ImageSource.gallery,
              ),
            ),
            const SizedBox(height: 12),
            _ActionTile(
              icon: Icons.auto_awesome,
              title: s.tryDemoImage,
              subtitle: s.analysisPreviewNote,
              emphasized: true,
              onTap: () {
                app.setActiveRequest(AnalysisRequest(
                  crop: app.selectedCrop,
                  isDemoImage: true,
                ));
                Navigator.of(context)
                    .pushNamed(AnalysisProgressScreen.routeName);
              },
            ),
            const SizedBox(height: 16),

            // -- Photo tips ---------------------------------------------
            AgriCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const _PhotoChecklistEntry(text: _tipOne),
                  const SizedBox(height: 8),
                  const _PhotoChecklistEntry(text: _tipTwo),
                  const SizedBox(height: 8),
                  const _PhotoChecklistEntry(text: _tipThree),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  static const String _tipOne =
      'One leaf fills the frame · natural daylight';
  static const String _tipTwo =
      'Leaf in sharp focus · plain color photo';
  static const String _tipThree =
      'AgriMind checks quality and verifies the crop before diagnosing';

  void _showCaptureSheet(
    BuildContext context,
    String title,
    String message,
    IconData icon,
    ImageSource source,
  ) {
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (BuildContext sheetContext) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 40, color: Colors.green.shade700),
              const SizedBox(height: 12),
              Text(
                title,
                style: Theme.of(sheetContext)
                    .textTheme
                    .titleLarge
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              Text(
                message,
                style: Theme.of(sheetContext)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(color: Colors.black87, height: 1.4),
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  icon: Icon(source == ImageSource.camera
                      ? Icons.photo_camera
                      : Icons.photo_library),
                  label: Text(source == ImageSource.camera
                      ? 'Launch Camera'
                      : 'Open Gallery'),
                  onPressed: () {
                    Navigator.of(sheetContext).pop();
                    _pickImage(context, source);
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _showPlaceholderSheet(
    BuildContext context,
    String title,
    String message,
    IconData icon,
  ) {
    showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      builder: (BuildContext sheetContext) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 40, color: Colors.green.shade700),
              const SizedBox(height: 12),
              Text(
                title,
                style: Theme.of(sheetContext)
                    .textTheme
                    .titleLarge
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              Text(
                message,
                style: Theme.of(sheetContext)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(color: Colors.black87, height: 1.4),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ActionTile extends StatelessWidget {
  const _ActionTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
    this.emphasized = false,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;
  final bool emphasized;

  @override
  Widget build(BuildContext context) {
    return AgriCard(
      onTap: onTap,
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: emphasized ? AgriColors.primary : AgriColors.softGreen,
              shape: BoxShape.circle,
            ),
            child: Icon(
              icon,
              color: emphasized ? Colors.white : AgriColors.heading,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: emphasized ? AgriColors.heading : Colors.black87,
                      ),
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
          const SizedBox(width: 8),
          const Icon(Icons.chevron_right, color: Colors.black38),
        ],
      ),
    );
  }
}

class _PhotoChecklistEntry extends StatelessWidget {
  const _PhotoChecklistEntry({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.check_circle_outline,
            size: 18, color: Colors.green.shade600),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            text,
            style: Theme.of(context)
                .textTheme
                .bodySmall
                ?.copyWith(color: Colors.black87),
          ),
        ),
      ],
    );
  }
}
