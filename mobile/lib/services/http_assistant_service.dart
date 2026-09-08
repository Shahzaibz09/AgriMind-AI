import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/assistant.dart';
import '../models/language.dart';
import 'api_config.dart';
import 'assistant_service.dart';

/// Real HTTP farmer assistant service that forwards agricultural
/// questions to the AgriMind AI FastAPI backend and receives structured
/// trilingual guidance.
class HttpAssistantService implements AssistantService {
  HttpAssistantService({http.Client? client, this.baseUrl})
      : _client = client ?? http.Client();

  final http.Client _client;
  final String? baseUrl;

  String get effectiveBaseUrl => baseUrl ?? ApiConfig.baseUrl;

  @override
  Future<AssistantAnswer> ask(AssistantQuestion question) async {
    try {
      final uri = Uri.parse('$effectiveBaseUrl/api/v1/assistant');
      final response = await _client
          .post(
            uri,
            headers: <String, String>{'Content-Type': 'application/json'},
            body: json.encode(<String, dynamic>{
              'crop': question.crop?.name ?? 'tomato',
              'predicted_disease': question.disease,
              'question': question.text,
              'language': question.language.code,
            }),
          )
          .timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final Map<String, dynamic> data =
            json.decode(utf8.decode(response.bodyBytes))
                as Map<String, dynamic>;
        return AssistantAnswer.fromJson(data);
      } else {
        return AssistantAnswer(
          text:
              'Guidance will appear here once the analysis backend is connected.',
          isPlaceholder: true,
          status: 'backend_offline',
        );
      }
    } catch (e) {
      return AssistantAnswer(
        text:
            'Guidance will appear here once the analysis backend is connected.',
        isPlaceholder: true,
        status: 'connection_error',
      );
    }
  }
}
