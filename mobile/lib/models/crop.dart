/// Crops supported by AgriMind AI.
///
/// Mirrors the crop options of the existing Streamlit web app
/// (Tomato, Potato, Apple) so the mobile MVP stays consistent with the
/// product.  Metadata is presentation-only.
enum Crop { tomato, potato, apple }

extension CropInfo on Crop {
  /// Display name (English label shared with the web app).
  String get label {
    return switch (this) {
      Crop.tomato => 'Tomato',
      Crop.potato => 'Potato',
      Crop.apple => 'Apple',
    };
  }

  /// Emoji used in the crop tiles.
  String get emoji {
    return switch (this) {
      Crop.tomato => '🍅',
      Crop.potato => '🥔',
      Crop.apple => '🍎',
    };
  }

  /// Short helper line shown under the crop name.
  String get tagline {
    return switch (this) {
      Crop.tomato => 'Leaf disease screening for tomato plants',
      Crop.potato => 'Leaf disease screening for potato plants',
      Crop.apple => 'Leaf disease screening for apple trees',
    };
  }
}
