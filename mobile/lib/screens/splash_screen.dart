import 'package:flutter/material.dart';

import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import '../widgets/brand.dart';
import 'home_screen.dart';

/// Splash screen — brand mark, name, tagline, then a short transition
/// to the Home screen.
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  static const String routeName = '/';

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future<void>.delayed(const Duration(milliseconds: 1800), _goHome);
  }

  void _goHome() {
    if (!mounted) return;
    Navigator.of(context).pushReplacementNamed(HomeScreen.routeName);
  }

  @override
  Widget build(BuildContext context) {
    final s = context.strings;
    return Scaffold(
      backgroundColor: AgriColors.scaffoldBackground,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const BrandMark(size: 104, iconSize: 52),
            const SizedBox(height: 24),
            Text(
              s.appName,
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    color: AgriColors.heading,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.5,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              s.tagline,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.black54,
                  ),
            ),
            const SizedBox(height: 32),
            const SizedBox(
              width: 24,
              height: 24,
              child: CircularProgressIndicator(strokeWidth: 2.5),
            ),
          ],
        ),
      ),
    );
  }
}
