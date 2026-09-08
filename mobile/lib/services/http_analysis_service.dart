import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/analysis.dart';
import 'analysis_service.dart';
import 'api_config.dart';

/// Real HTTP leaf analysis service that sends the leaf photo to the
/// AgriMind AI FastAPI backend and receives the guarded AI diagnosis
/// and Grad-CAM overlay.
class HttpAnalysisService implements AnalysisService {
  HttpAnalysisService({http.Client? client, this.baseUrl})
      : _client = client ?? http.Client();

  final http.Client _client;
  final String? baseUrl;

  String get effectiveBaseUrl => baseUrl ?? ApiConfig.baseUrl;

  @override
  Future<AnalysisResult> analyzeLeaf(AnalysisRequest request) async {
    // If it's a pure demo preview without image bytes, return the placeholder result
    if (request.isDemoImage &&
        (request.imageBytes == null || request.imageBytes!.isEmpty)) {
      return AnalysisResult.placeholderFor(request.crop);
    }

    if (request.imageBytes == null || request.imageBytes!.isEmpty) {
      return AnalysisResult.error(
        request.crop,
        'No image provided for analysis.',
      );
    }

    try {
      final uri = Uri.parse('$effectiveBaseUrl/api/v1/analyze');
      final httpRequest = http.MultipartRequest('POST', uri);

      httpRequest.fields['crop'] = request.crop.name;
      httpRequest.files.add(
        http.MultipartFile.fromBytes(
          'image',
          request.imageBytes!,
          filename: request.imageName ?? 'leaf.jpg',
        ),
      );

      final streamedResponse = await _client.send(httpRequest).timeout(
            const Duration(seconds: 45),
          );
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final Map<String, dynamic> data =
            json.decode(response.body) as Map<String, dynamic>;
        return AnalysisResult.fromJson(data, request.crop);
      } else {
        return AnalysisResult.error(
          request.crop,
          'Backend returned status code ${response.statusCode}: ${response.body}',
        );
      }
    } catch (e) {
      return AnalysisResult.error(
        request.crop,
        'Cannot connect to backend ($effectiveBaseUrl): $e',
      );
    }
  }
}
