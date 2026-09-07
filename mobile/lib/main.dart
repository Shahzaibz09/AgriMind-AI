import 'package:flutter/material.dart';

import 'models/language.dart';
import 'screens/analysis_progress_screen.dart';
import 'screens/assistant_screen.dart';
import 'screens/crop_selection_screen.dart';
import 'screens/home_screen.dart';
import 'screens/language_screen.dart';
import 'screens/leaf_analysis_screen.dart';
import 'screens/result_screen.dart';
import 'screens/splash_screen.dart';
import 'state/app_scope.dart';
import 'state/app_state.dart';
import 'theme/app_theme.dart';

void main() {
  runApp(const AgriMindApp());
}

/// AgriMind AI — mobile MVP (Step 1: UI foundation).
///
/// Navigation map:
///
/// ```text
/// Splash → Home ┬→ Crop Selection → Leaf Analysis → Analysis → Result
///               ├→ Farmer Assistant
///               └→ Language Selection
/// ```
class AgriMindApp extends StatefulWidget {
  const AgriMindApp({super.key});

  @override
  State<AgriMindApp> createState() => _AgriMindAppState();
}

class _AgriMindAppState extends State<AgriMindApp> {
  final AppState _appState = AppState();

  @override
  void dispose() {
    _appState.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _appState,
      builder: (BuildContext context, _) {
        return AppScope(
          notifier: _appState,
          child: MaterialApp(
            title: 'AgriMind AI',
            debugShowCheckedModeBanner: false,
            theme: AgriTheme.light,
            // Urdu (اردو) is right-to-left; the whole app mirrors.
            builder: (BuildContext context, Widget? child) {
              return Directionality(
                textDirection: _appState.language.textDirection,
                child: child!,
              );
            },
            initialRoute: SplashScreen.routeName,
            routes: <String, WidgetBuilder>{
              SplashScreen.routeName: (_) => const SplashScreen(),
              HomeScreen.routeName: (_) => const HomeScreen(),
              CropSelectionScreen.routeName: (_) => const CropSelectionScreen(),
              LeafAnalysisScreen.routeName: (_) => const LeafAnalysisScreen(),
              AnalysisProgressScreen.routeName: (_) =>
                  const AnalysisProgressScreen(),
              ResultScreen.routeName: (_) => const ResultScreen(),
              AssistantScreen.routeName: (_) => const AssistantScreen(),
              LanguageScreen.routeName: (_) => const LanguageScreen(),
            },
          ),
        );
      },
    );
  }
}
