import '../models/language.dart';

/// UI strings for AgriMind AI in the three supported languages.
///
/// This is the language-selection *architecture* only: it covers app
/// chrome (titles, buttons, step names).  Agricultural knowledge content
/// stays in the existing knowledge base and is not duplicated here.
class AppStrings {
  const AppStrings._(this._values);

  final Map<String, String> _values;

  String get appName => _v('appName');
  String get tagline => _v('tagline');
  String get homeHeadline => _v('homeHeadline');
  String get homeDescription => _v('homeDescription');
  String get analyzeCrop => _v('analyzeCrop');
  String get askAgriMind => _v('askAgriMind');
  String get howItWorks => _v('howItWorks');
  String get language => _v('language');
  String get aboutTitle => _v('aboutTitle');
  String get aboutBody => _v('aboutBody');
  String get selectYourCrop => _v('selectYourCrop');
  String get selected => _v('selected');
  String get continueLabel => _v('continueLabel');
  String get uploadLeafPhoto => _v('uploadLeafPhoto');
  String get takePhoto => _v('takePhoto');
  String get chooseFromGallery => _v('chooseFromGallery');
  String get tryDemoImage => _v('tryDemoImage');
  String get takePhotoHelp => _v('takePhotoHelp');
  String get analyzingLeaf => _v('analyzingLeaf');
  String get qualityCheck => _v('qualityCheck');
  String get cropVerification => _v('cropVerification');
  String get diseaseAnalysis => _v('diseaseAnalysis');
  String get analysisPreviewNote => _v('analysisPreviewNote');
  String get viewResult => _v('viewResult');
  String get diagnosisTitle => _v('diagnosisTitle');
  String get cropLabel => _v('cropLabel');
  String get diseaseLabel => _v('diseaseLabel');
  String get confidenceLabel => _v('confidenceLabel');
  String get statusLabel => _v('statusLabel');
  String get recommendedNextStep => _v('recommendedNextStep');
  String get analyzeAnotherLeaf => _v('analyzeAnotherLeaf');
  String get gradcamTitle => _v('gradcamTitle');
  String get lessInfluence => _v('lessInfluence');
  String get moreInfluence => _v('moreInfluence');
  String get placeholderNotice => _v('placeholderNotice');
  String get notAvailableYet => _v('notAvailableYet');
  String get yourQuestion => _v('yourQuestion');
  String get ask => _v('ask');
  String get questionSymptoms => _v('questionSymptoms');
  String get questionManage => _v('questionManage');
  String get questionPrevent => _v('questionPrevent');
  String get guidancePending => _v('guidancePending');
  String get appLanguage => _v('appLanguage');
  String get languageChanged => _v('languageChanged');

  String _v(String key) => _values[key] ?? _englishValues[key] ?? key;

  /// Returns the strings for [language] (English fallback for any key
  /// missing from a translation).
  static AppStrings forLanguage(AppLanguage language) {
    return switch (language) {
      AppLanguage.english => const AppStrings._(_englishValues),
      AppLanguage.urdu => const AppStrings._(_urduValues),
      AppLanguage.romanUrdu => const AppStrings._(_romanUrduValues),
    };
  }

  static const Map<String, String> _englishValues = {
    'appName': 'AgriMind AI',
    'tagline': 'AI-powered Crop Health Intelligence',
    'homeHeadline': 'Check your crop leaf health in seconds',
    'homeDescription':
        'Photograph a single leaf and AgriMind AI checks photo quality, '
        'verifies the crop, and explains the diagnosis — with guidance in '
        'your language.',
    'analyzeCrop': 'Analyze Crop',
    'askAgriMind': 'Ask AgriMind',
    'howItWorks': 'How it works',
    'language': 'Language',
    'aboutTitle': 'About AgriMind AI',
    'aboutBody':
        'AgriMind AI is an AI-assisted screening tool for crop leaf '
        'diseases, built for farmers. It is not a replacement for '
        'agricultural experts — always confirm important decisions with '
        'your local extension office.',
    'selectYourCrop': 'Select your crop',
    'selected': 'Selected',
    'continueLabel': 'Continue',
    'uploadLeafPhoto': 'Upload a leaf photo',
    'takePhoto': 'Take Photo',
    'chooseFromGallery': 'Choose from Gallery',
    'tryDemoImage': 'Try Demo Image',
    'takePhotoHelp': 'Camera and gallery connect in the next step.',
    'analyzingLeaf': 'Analyzing your leaf',
    'qualityCheck': 'Image quality check',
    'cropVerification': 'Crop verification',
    'diseaseAnalysis': 'Disease analysis',
    'analysisPreviewNote':
        'Preview only — the real checks run once the backend is connected.',
    'viewResult': 'View result',
    'diagnosisTitle': 'Diagnosis',
    'cropLabel': 'Crop',
    'diseaseLabel': 'Disease',
    'confidenceLabel': 'Confidence',
    'statusLabel': 'Status',
    'recommendedNextStep': 'Recommended next step',
    'analyzeAnotherLeaf': 'Analyze Another Leaf',
    'gradcamTitle': 'Why this prediction? (Grad-CAM)',
    'lessInfluence': 'Less influence',
    'moreInfluence': 'More influence',
    'placeholderNotice':
        'Demo preview — not a real diagnosis. Results arrive once the '
        'analysis backend is connected.',
    'notAvailableYet': 'Not available yet',
    'yourQuestion': 'Your question',
    'ask': 'Ask',
    'questionSymptoms': 'What are the symptoms?',
    'questionManage': 'How can I manage it?',
    'questionPrevent': 'How can I prevent it?',
    'guidancePending':
        'Guidance will appear here once the analysis backend is connected.',
    'appLanguage': 'App language',
    'languageChanged': 'Language updated',
  };

  static const Map<String, String> _urduValues = {
    'tagline': 'اے آئی سے چلنے والی فصل صحت کی معلومات',
    'homeHeadline': 'سیکنڈوں میں فصل کے پتے کی صحت جانچیں',
    'homeDescription':
        'ایک پتے کی تصویر لیں — AgriMind AI تصویر کا معیار جانچتا ہے، فصل '
        'کی تصدیق کرتا ہے، اور تشخیص وضاحت کے ساتھ پیش کرتا ہے۔ رہنمائی آپ '
        'کی زبان میں ملتی ہے۔',
    'analyzeCrop': 'فصل کا تجزیہ کریں',
    'askAgriMind': 'AgriMind سے پوچھیں',
    'howItWorks': 'یہ کیسے کام کرتا ہے',
    'language': 'زبان',
    'aboutTitle': 'AgriMind AI کے بارے میں',
    'aboutBody':
        'AgriMind AI فصل کے پتوں کی بیماریوں کے لیے اے آئی معاون جانچ کا '
        'آلہ ہے۔ یہ زرعی ماہرین کا متبادل نہیں — اہم فیصلوں کی تصدیق اپنے '
        'مقامی زرعی دفتر سے کریں۔',
    'selectYourCrop': 'اپنی فصل منتخب کریں',
    'selected': 'منتخب',
    'continueLabel': 'جاری رکھیں',
    'uploadLeafPhoto': 'پتے کی تصویر اپ لوڈ کریں',
    'takePhoto': 'تصویر لیں',
    'chooseFromGallery': 'گیلری سے منتخب کریں',
    'tryDemoImage': 'ڈیمو تصویر آزمائیں',
    'takePhotoHelp': 'کیمرہ اور گیلری اگلے مرحلے میں جڑیں گی۔',
    'analyzingLeaf': 'پتے کا تجزیہ ہو رہا ہے',
    'qualityCheck': 'تصویر کا معیار جانچ',
    'cropVerification': 'فصل کی تصدیق',
    'diseaseAnalysis': 'بیماری کا تجزیہ',
    'analysisPreviewNote': 'صرف پیش نظارہ — اصل جانچ سسٹم جڑنے کے بعد ہوگی۔',
    'viewResult': 'نتیجہ دیکھیں',
    'diagnosisTitle': 'تشخیص',
    'cropLabel': 'فصل',
    'diseaseLabel': 'بیماری',
    'confidenceLabel': 'اعتماد',
    'statusLabel': 'حیثیت',
    'recommendedNextStep': 'تجویز کردہ اگلا قدم',
    'analyzeAnotherLeaf': 'دوسرا پتہ تجزیہ کریں',
    'gradcamTitle': 'یہ تشخیص کیوں؟ (Grad-CAM)',
    'lessInfluence': 'کم اثر',
    'moreInfluence': 'زیادہ اثر',
    'placeholderNotice': 'ڈیمو پیش نظارہ — یہ حقیقی تشخیص نہیں۔',
    'notAvailableYet': 'ابھی دستیاب نہیں',
    'yourQuestion': 'آپ کا سوال',
    'ask': 'پوچھیں',
    'questionSymptoms': 'علامات کیا ہیں؟',
    'questionManage': 'میں اس کا علاج کیسے کروں؟',
    'questionPrevent': 'میں اسے کیسے روک سکتا/سکتی ہوں؟',
    'guidancePending': 'تجزیہ کا سسٹم جڑنے کے بعد یہاں رہنمائی دکھائی دے گی۔',
    'appLanguage': 'ایپ کی زبان',
    'languageChanged': 'زبان تبدیل ہو گئی',
  };

  static const Map<String, String> _romanUrduValues = {
    'tagline': 'AI se chalne wali fasal sehat ki maloomat',
    'homeHeadline': 'Seconds mein fasal ke patte ki sehat check karein',
    'homeDescription':
        'Aik patte ki tasveer lein — AgriMind AI tasveer ka mayaar check '
        'karta hai, fasal ki tasdeeq karta hai, aur tajziya wazahat ke saath '
        'dikhata hai. Rehnumai aap ki zaban mein milti hai.',
    'analyzeCrop': 'Fasal ka tajziya karein',
    'askAgriMind': 'AgriMind se poochein',
    'howItWorks': 'Yeh kaise kaam karta hai',
    'language': 'Zaban',
    'aboutTitle': 'AgriMind AI ke baare mein',
    'aboutBody':
        'AgriMind AI fasal ke pattiyon ki bimariyon ke liye AI madadgar '
        'jaanch ka aala hai. Yeh zaraee mahireen ka mutabadil nahi — ahem '
        'faislon ki tasdeeq apne local zaraee daftarse karein.',
    'selectYourCrop': 'Apni fasal muntakhib karein',
    'selected': 'Muntakhib',
    'continueLabel': 'Jaari rakhein',
    'uploadLeafPhoto': 'Patte ki tasveer upload karein',
    'takePhoto': 'Tasveer lein',
    'chooseFromGallery': 'Gallery se muntakhib karein',
    'tryDemoImage': 'Demo tasveer azmayein',
    'takePhotoHelp': 'Camera aur gallery aglay marhalay mein judengein.',
    'analyzingLeaf': 'Patte ka tajziya ho raha hai',
    'qualityCheck': 'Tasveer ka mayaar check',
    'cropVerification': 'Fasal ki tasdeeq',
    'diseaseAnalysis': 'Bimari ka tajziya',
    'analysisPreviewNote':
        'Sirf preview — asli checks system judnay ke baad chalengein.',
    'viewResult': 'Nateeja dekhein',
    'diagnosisTitle': 'Tajziya',
    'cropLabel': 'Fasal',
    'diseaseLabel': 'Bimari',
    'confidenceLabel': 'Aitmaad',
    'statusLabel': 'Haisiyat',
    'recommendedNextStep': 'Tajveez karda agla qadam',
    'analyzeAnotherLeaf': 'Doosra patta tajziya karein',
    'gradcamTitle': 'Yeh tajziya kyun? (Grad-CAM)',
    'lessInfluence': 'Kam asar',
    'moreInfluence': 'Zyada asar',
    'placeholderNotice': 'Demo preview — asli tajziya nahi.',
    'notAvailableYet': 'Abhi dastyaab nahi',
    'yourQuestion': 'Aap ka sawal',
    'ask': 'Poochein',
    'questionSymptoms': 'Alamaat kya hain?',
    'questionManage': 'Main iska ilaj kaise karoon?',
    'questionPrevent': 'Main ise kaise rok sakta hoon?',
    'guidancePending': 'Tajziya ka system judnay ke baad yahan rehnumai aayegi.',
    'appLanguage': 'App ki zaban',
    'languageChanged': 'Zaban tabdeel ho gayi',
  };
}
