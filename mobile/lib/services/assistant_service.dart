import '../models/assistant.dart';

/// Farmer Assistant contract for the AgriMind AI mobile app.
///
/// Step 2 will provide an implementation that forwards questions to the
/// planned FastAPI backend, which reuses the existing deterministic
/// trilingual Farmer Assistant.  The interface mirrors the web app's
/// concept: the question, selected crop, and latest diagnosis provide
/// the answer's context.
abstract interface class AssistantService {
  Future<AssistantAnswer> ask(AssistantQuestion question);
}

/// Step 1 placeholder implementation.
///
/// Returns a clearly marked "guidance pending" message in the chosen
/// language.  No agricultural advice is generated on the device.
class DemoAssistantService implements AssistantService {
  const DemoAssistantService();

  @override
  Future<AssistantAnswer> ask(AssistantQuestion question) async {
    await Future<void>.delayed(const Duration(milliseconds: 400));
    return AssistantAnswer(
      text: switch (question.language) {
        _ => 'Guidance will appear here once the analysis backend is '
            'connected.',
      },
      isPlaceholder: true,
    );
  }
}
