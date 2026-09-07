import 'package:flutter_test/flutter_test.dart';

import 'package:agrimind_ai/models/analysis.dart';
import 'package:agrimind_ai/models/assistant.dart';
import 'package:agrimind_ai/models/crop.dart';
import 'package:agrimind_ai/models/language.dart';
import 'package:agrimind_ai/services/analysis_service.dart';
import 'package:agrimind_ai/services/assistant_service.dart';

void main() {
  group('DemoAnalysisService', () {
    test('returns a clearly marked placeholder, never fake AI data', () async {
      const DemoAnalysisService service = DemoAnalysisService(
        simulatedDelay: Duration.zero,
      );

      final AnalysisResult result = await service.analyzeLeaf(
        AnalysisRequest(crop: Crop.tomato, isDemoImage: true),
      );

      expect(result.isPlaceholder, isTrue);
      expect(result.disease, isNull);
      expect(result.confidence, isNull);
      expect(result.crop, Crop.tomato);
    });
  });

  group('DemoAssistantService', () {
    test('answers with a placeholder — no advice invented on device',
        () async {
      const DemoAssistantService service = DemoAssistantService();

      final AssistantAnswer answer = await service.ask(
        AssistantQuestion(
          text: 'What are the symptoms?',
          language: AppLanguage.english,
          crop: Crop.tomato,
        ),
      );

      expect(answer.isPlaceholder, isTrue);
      expect(answer.text, contains('backend'));
      // Must not contain fabricated guidance.
      expect(answer.text.contains('blight'), isFalse);
    });
  });
}
