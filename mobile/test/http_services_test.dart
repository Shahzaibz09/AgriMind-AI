import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:agrimind_ai/models/analysis.dart';
import 'package:agrimind_ai/models/assistant.dart';
import 'package:agrimind_ai/models/crop.dart';
import 'package:agrimind_ai/models/language.dart';
import 'package:agrimind_ai/services/http_analysis_service.dart';
import 'package:agrimind_ai/services/http_assistant_service.dart';

void main() {
  group('HttpAnalysisService', () {
    test('successfully parses AI diagnosis and Grad-CAM overlay', () async {
      final client = MockClient((request) async {
        expect(request.url.path, '/api/v1/analyze');
        return http.Response(
          json.encode({
            'status': 'success',
            'crop': 'tomato',
            'crop_label': 'Tomato',
            'predicted_disease': 'Early Blight',
            'confidence': 0.965,
            'probabilities': {
              'Healthy': 0.01,
              'Early Blight': 0.965,
              'Late Blight': 0.025,
            },
            'gradcam_overlay_base64': 'data:image/jpeg;base64,fakeimagebytes',
            'status_label': 'Action Recommended',
            'description': 'Early blight description',
            'recommended_next_step': 'Isolate affected plants',
            'cautions': ['Close margin call'],
          }),
          200,
        );
      });

      final service =
          HttpAnalysisService(client: client, baseUrl: 'http://test-server');
      final result = await service.analyzeLeaf(
        AnalysisRequest(
          crop: Crop.tomato,
          imageBytes: Uint8List.fromList([1, 2, 3]),
          imageName: 'test.jpg',
        ),
      );

      expect(result.isPlaceholder, isFalse);
      expect(result.disease, 'Early Blight');
      expect(result.confidence, 0.965);
      expect(result.statusLabel, 'Action Recommended');
      expect(result.gradcamOverlayBase64, contains('fakeimagebytes'));
      expect(result.cautions, contains('Close margin call'));
    });

    test('handles quality rejection gracefully', () async {
      final client = MockClient((request) async {
        return http.Response(
          json.encode({
            'status': 'quality_rejected',
            'crop': 'tomato',
            'crop_label': 'Tomato',
            'message': 'No plant-like green content found.',
            'reasons': ['no_plant_material'],
          }),
          200,
        );
      });

      final service =
          HttpAnalysisService(client: client, baseUrl: 'http://test-server');
      final result = await service.analyzeLeaf(
        AnalysisRequest(
          crop: Crop.tomato,
          imageBytes: Uint8List.fromList([1, 2, 3]),
        ),
      );

      expect(result.apiStatus, 'quality_rejected');
      expect(result.errorMessage, contains('No plant-like green content'));
    });

    test('handles crop mismatch gracefully', () async {
      final client = MockClient((request) async {
        return http.Response(
          json.encode({
            'status': 'crop_mismatch',
            'crop': 'tomato',
            'crop_label': 'Tomato',
            'detected_crop': 'Apple',
            'confidence': 0.98,
            'message':
                'This image appears to be an Apple leaf, but Tomato was selected.',
          }),
          200,
        );
      });

      final service =
          HttpAnalysisService(client: client, baseUrl: 'http://test-server');
      final result = await service.analyzeLeaf(
        AnalysisRequest(
          crop: Crop.tomato,
          imageBytes: Uint8List.fromList([1, 2, 3]),
        ),
      );

      expect(result.apiStatus, 'crop_mismatch');
      expect(result.detectedCrop, 'Apple');
      expect(result.errorMessage, contains('Apple leaf'));
    });

    test('handles connection failure gracefully', () async {
      final client = MockClient((request) async {
        throw Exception('Network unreachable');
      });

      final service =
          HttpAnalysisService(client: client, baseUrl: 'http://test-server');
      final result = await service.analyzeLeaf(
        AnalysisRequest(
          crop: Crop.tomato,
          imageBytes: Uint8List.fromList([1, 2, 3]),
        ),
      );

      expect(result.apiStatus, 'error');
      expect(result.errorMessage, contains('Network unreachable'));
    });
  });

  group('HttpAssistantService', () {
    test('sends question and receives structured trilingual guidance', () async {
      final client = MockClient((request) async {
        expect(request.url.path, '/api/v1/assistant');
        final body = json.decode(request.body) as Map<String, dynamic>;
        expect(body['crop'], 'tomato');
        expect(body['language'], 'ur');

        return http.Response(
          json.encode({
            'status': 'ok',
            'language': 'ur',
            'response': 'عام علامات: پتوں پر بھورے دھبے',
            'disease': 'Early Blight',
            'confidence': 0.95,
          }),
          200,
          headers: {'content-type': 'application/json; charset=utf-8'},
        );
      });

      final service =
          HttpAssistantService(client: client, baseUrl: 'http://test-server');
      final answer = await service.ask(
        const AssistantQuestion(
          text: 'علامات کیا ہیں؟',
          language: AppLanguage.urdu,
          crop: Crop.tomato,
          disease: 'Early Blight',
        ),
      );

      expect(answer.isPlaceholder, isFalse);
      expect(answer.text, contains('عام علامات'));
    });
  });
}
