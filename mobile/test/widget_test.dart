import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:agrimind_ai/main.dart';

void main() {
  testWidgets('splash shows branding then transitions to Home',
      (WidgetTester tester) async {
    await tester.pumpWidget(const AgriMindApp());

    // Splash content is visible immediately.
    expect(find.text('AgriMind AI'), findsOneWidget);
    expect(find.text('AI-powered Crop Health Intelligence'), findsOneWidget);
    expect(find.byType(CircularProgressIndicator), findsOneWidget);

    // After the splash delay the Home screen replaces it.
    await tester.pump(const Duration(milliseconds: 1900));
    await tester.pumpAndSettle();

    expect(find.text('Analyze Crop'), findsOneWidget);
    expect(find.text('Ask AgriMind'), findsOneWidget);
    expect(find.text('How it works'), findsOneWidget);

    // The About card sits below the fold — scroll it into view.
    await tester.scrollUntilVisible(find.text('About AgriMind AI'), 300);
    expect(find.text('About AgriMind AI'), findsOneWidget);
  });
}
