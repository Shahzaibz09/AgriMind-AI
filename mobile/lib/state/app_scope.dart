import 'package:flutter/widgets.dart';

import '../l10n/app_strings.dart';
import 'app_state.dart';

/// Exposes [AppState] to the widget tree.
///
/// Any widget below [AppScope] rebuilds automatically when the state
/// changes (crop selection, language switch, analysis updates).
class AppScope extends InheritedNotifier<AppState> {
  const AppScope({
    super.key,
    required AppState notifier,
    required super.child,
  }) : super(notifier: notifier);

  /// The closest [AppState] instance.
  static AppState of(BuildContext context) {
    final AppScope? scope =
        context.dependOnInheritedWidgetOfExactType<AppScope>();
    assert(scope != null, 'AppScope not found — wrap the app in AgriMindApp');
    return scope!.notifier!;
  }
}

/// Convenience access to localized strings.
extension ContextStrings on BuildContext {
  AppStrings get strings => AppScope.of(this).strings;
}
