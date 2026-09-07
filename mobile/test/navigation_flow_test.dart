import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:agrimind_ai/main.dart';

/// Helper pumping the app straight to the Home screen.
Future<void> _pumpToHome(WidgetTester tester) async {
  await tester.pumpWidget(const AgriMindApp());
  await tester.pump(const Duration(milliseconds: 1900));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets(
      'full demo flow: crop selection → demo image → analysis → '
      'placeholder result → reset', (WidgetTester tester) async {
    await _pumpToHome(tester);

    // -- Home → Crop Selection -----------------------------------------
    await tester.tap(find.text('Analyze Crop'));
    await tester.pumpAndSettle();
    expect(find.text('Select your crop'), findsOneWidget);
    // Tomato is the default (like the web app).
    expect(find.text('Tomato'), findsOneWidget);

    // -- Select Potato, clearly marked ----------------------------------
    await tester.tap(find.text('Potato'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Continue'));
    await tester.pumpAndSettle();

    // -- Leaf Analysis screen -------------------------------------------
    expect(find.text('Upload a leaf photo'), findsOneWidget);
    expect(find.text('Take Photo'), findsOneWidget);
    expect(find.text('Choose from Gallery'), findsOneWidget);
    expect(find.text('Try Demo Image'), findsOneWidget);
    // Selected crop is visible on the analysis screen.
    expect(find.text('Selected: Potato'), findsOneWidget);

    // -- Placeholder actions explain the next step ----------------------
    await tester.tap(find.text('Take Photo'));
    await tester.pumpAndSettle();
    // The bottom sheet repeats the "next step" note (also used as tile
    // subtitles), so several instances are expected.
    expect(find.text('Camera and gallery connect in the next step.'),
        findsAtLeastNWidgets(3));
    // Dismiss the sheet by tapping the modal barrier.
    await tester.tapAt(const Offset(20, 20));
    await tester.pumpAndSettle();

    // -- Run the demo analysis flow --------------------------------------
    // (No pumpAndSettle here: the active-stage spinner animates forever.)
    await tester.tap(find.text('Try Demo Image'));
    await tester.pump(); // process the tap and the route push
    await tester.pump(const Duration(milliseconds: 400)); // enter transition
    expect(find.text('Image quality check'), findsOneWidget);
    expect(find.text('Crop verification'), findsOneWidget);
    expect(find.text('Disease analysis'), findsOneWidget);

    // Let the three preview stages and the service call complete.
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump(const Duration(milliseconds: 900));
    await tester.pump(const Duration(milliseconds: 300));

    // -- Result screen is clearly marked as placeholder ------------------
    await tester.tap(find.text('View result'));
    await tester.pumpAndSettle();

    expect(find.text('Diagnosis'), findsOneWidget);
    expect(find.text('Demo preview — not a real diagnosis. Results arrive '
        'once the analysis backend is connected.'), findsOneWidget);
    expect(find.text('Not available yet'), findsAtLeastNWidgets(4));

    await tester
        .scrollUntilVisible(find.text('Why this prediction? (Grad-CAM)'), 200);
    expect(find.text('Why this prediction? (Grad-CAM)'), findsOneWidget);

    // -- Analyze Another Leaf resets to the upload step ------------------
    await tester.scrollUntilVisible(find.text('Analyze Another Leaf'), 300);
    await tester.tap(find.text('Analyze Another Leaf'));
    await tester.pumpAndSettle();

    expect(find.text('Upload a leaf photo'), findsOneWidget);
    expect(find.text('Selected: Potato'), findsOneWidget);
  });

  testWidgets('Farmer Assistant quick questions produce placeholder replies',
      (WidgetTester tester) async {
    await _pumpToHome(tester);

    await tester.tap(find.text('Ask AgriMind'));
    await tester.pumpAndSettle();

    // Quick-pick chips exist (same intents as the web app).
    expect(find.text('What are the symptoms?'), findsOneWidget);
    expect(find.text('How can I manage it?'), findsOneWidget);
    expect(find.text('How can I prevent it?'), findsOneWidget);

    await tester.tap(find.text('What are the symptoms?'));
    await tester.pump(const Duration(milliseconds: 500));
    await tester.pumpAndSettle();

    // The question is echoed and a placeholder answer appears.
    expect(find.text('What are the symptoms?'), findsWidgets);
    expect(find.textContaining('Guidance will appear here'), findsOneWidget);
  });

  testWidgets('language selection switches the UI, including RTL for Urdu',
      (WidgetTester tester) async {
    await _pumpToHome(tester);

    // Open the language screen from the app bar.
    await tester.tap(find.byIcon(Icons.translate));
    await tester.pumpAndSettle();

    expect(find.text('English'), findsOneWidget);
    expect(find.text('اردو (Urdu)'), findsOneWidget);
    expect(find.text('Roman Urdu'), findsOneWidget);

    // -- Switch to Urdu ---------------------------------------------------
    await tester.tap(find.text('اردو (Urdu)'));
    await tester.pumpAndSettle();

    // The language screen itself switches to Urdu and mirrors to RTL.
    expect(find.text('ایپ کی زبان'), findsWidgets);
    expect(find.text('سیکنڈوں میں فصل کے پتے کی صحت جانچیں'), findsOneWidget);
    final BuildContext urduContext =
        tester.element(find.text('ایپ کی زبان').first);
    expect(Directionality.of(urduContext), TextDirection.rtl);

    // Back on Home the action labels are Urdu too.
    await tester.pageBack();
    await tester.pumpAndSettle();
    expect(find.text('فصل کا تجزیہ کریں'), findsOneWidget);
    expect(find.text('AgriMind سے پوچھیں'), findsOneWidget);
    final BuildContext homeUrduContext =
        tester.element(find.text('فصل کا تجزیہ کریں'));
    expect(Directionality.of(homeUrduContext), TextDirection.rtl);

    // -- Switch to Roman Urdu ---------------------------------------------
    await tester.tap(find.byIcon(Icons.translate));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Roman Urdu'));
    await tester.pumpAndSettle();

    expect(find.text('App ki zaban'), findsWidgets);

    await tester.pageBack();
    await tester.pumpAndSettle();
    expect(find.text('Fasal ka tajziya karein'), findsOneWidget);
    final BuildContext homeRomanContext =
        tester.element(find.text('Fasal ka tajziya karein'));
    expect(Directionality.of(homeRomanContext), TextDirection.ltr);
  });
}
