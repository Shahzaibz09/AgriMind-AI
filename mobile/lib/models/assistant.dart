import 'crop.dart';
import 'language.dart';

/// A farmer's question for the assistant.
///
/// In the final architecture the question, selected crop and latest
/// diagnosis travel to the FastAPI backend, which reuses the existing
/// deterministic Farmer Assistant logic.
class AssistantQuestion {
  const AssistantQuestion({
    required this.text,
    required this.language,
    this.crop,
    this.disease,
  });

  final String text;
  final AppLanguage language;
  final Crop? crop;
  final String? disease;
}

/// The assistant's answer.
///
/// Step 1 only produces placeholder answers ([isPlaceholder] `true`) —
/// no agricultural advice is invented on the device.
class AssistantAnswer {
  const AssistantAnswer({required this.text, this.isPlaceholder = false});

  final String text;
  final bool isPlaceholder;
}
