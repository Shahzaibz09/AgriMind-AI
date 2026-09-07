import 'package:flutter/widgets.dart' show TextDirection;
import 'package:flutter_test/flutter_test.dart';

import 'package:agrimind_ai/models/analysis.dart';
import 'package:agrimind_ai/models/crop.dart';
import 'package:agrimind_ai/models/language.dart';
import 'package:agrimind_ai/l10n/app_strings.dart';

void main() {
  group('Crop', () {
    test('supports exactly the web app crops in demo order', () {
      expect(Crop.values.map((c) => c.label).toList(),
          ['Tomato', 'Potato', 'Apple']);
    });

    test('metadata is presentation-only and complete', () {
      for (final Crop crop in Crop.values) {
        expect(crop.label, isNotEmpty);
        expect(crop.emoji, isNotEmpty);
        expect(crop.tagline, isNotEmpty);
      }
    });
  });

  group('AppLanguage', () {
    test('covers English, Urdu and Roman Urdu', () {
      expect(AppLanguage.values.length, 3);
      expect(AppLanguage.english.code, 'en');
      expect(AppLanguage.urdu.code, 'ur');
      expect(AppLanguage.romanUrdu.code, 'roman_ur');
    });

    test('Urdu is right-to-left, the others left-to-right', () {
      expect(AppLanguage.urdu.textDirection, TextDirection.rtl);
      expect(AppLanguage.english.textDirection, TextDirection.ltr);
      expect(AppLanguage.romanUrdu.textDirection, TextDirection.ltr);
    });
  });

  group('AppStrings', () {
    test('returns translated chrome strings for each language', () {
      final AppStrings en = AppStrings.forLanguage(AppLanguage.english);
      final AppStrings ur = AppStrings.forLanguage(AppLanguage.urdu);
      final AppStrings roman = AppStrings.forLanguage(AppLanguage.romanUrdu);

      expect(en.analyzeCrop, 'Analyze Crop');
      expect(ur.analyzeCrop, 'فصل کا تجزیہ کریں');
      expect(roman.analyzeCrop, 'Fasal ka tajziya karein');

      // Quick questions exist in every language (same intents as web).
      for (final AppStrings s in [en, ur, roman]) {
        expect(s.questionSymptoms, isNotEmpty);
        expect(s.questionManage, isNotEmpty);
        expect(s.questionPrevent, isNotEmpty);
      }
    });

    test('falls back to English for untranslated keys', () {
      final AppStrings strings = AppStrings.forLanguage(AppLanguage.urdu);
      // appName is brand identity — identical in every language.
      expect(strings.appName, 'AgriMind AI');
    });
  });

  group('AnalysisResult', () {
    test('placeholder result never fabricates diagnosis data', () {
      final AnalysisResult result =
          AnalysisResult.placeholderFor(Crop.potato);
      expect(result.crop, Crop.potato);
      expect(result.isPlaceholder, isTrue);
      expect(result.disease, isNull);
      expect(result.confidence, isNull);
      expect(result.recommendedNextStep, isNull);
    });
  });
}
