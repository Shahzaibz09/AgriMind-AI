import 'package:flutter/material.dart';

import '../models/assistant.dart';
import '../services/assistant_service.dart';
import '../state/app_scope.dart';
import '../theme/app_theme.dart';
import 'language_screen.dart';

/// Farmer Assistant screen — question input, quick-question chips and a
/// conversation area.
///
/// Answers come from an [AssistantService]; the Step 1 demo service
/// returns a clearly marked "guidance pending" reply (no advice is
/// invented on the device).
class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key});

  static const String routeName = '/assistant';

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final AssistantService _service = const DemoAssistantService();
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<_ConversationEntry> _entries = <_ConversationEntry>[];
  bool _busy = false;

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _ask(String question) async {
    if (question.trim().isEmpty) return;
    final app = AppScope.of(context);
    setState(() {
      _busy = true;
      _entries.add(_ConversationEntry.question(question.trim()));
    });
    _controller.clear();
    _scrollToBottom();

    final AssistantAnswer answer = await _service.ask(
      AssistantQuestion(
        text: question,
        language: app.language,
        crop: app.selectedCrop,
        disease: app.lastResult?.disease,
      ),
    );
    if (!mounted) return;
    setState(() {
      _busy = false;
      _entries.add(_ConversationEntry.answer(answer));
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = context.strings;

    return Scaffold(
      appBar: AppBar(
        title: Text(s.askAgriMind),
        actions: [
          IconButton(
            tooltip: s.language,
            icon: const Icon(Icons.translate),
            onPressed: () =>
                Navigator.of(context).pushNamed(LanguageScreen.routeName),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // -- Conversation area --------------------------------------
            Expanded(
              child: _entries.isEmpty
                  ? Center(
                      child: Padding(
                        padding: const EdgeInsets.all(32),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.forum_outlined,
                                size: 44, color: Colors.green.shade300),
                            const SizedBox(height: 12),
                            Text(
                              s.guidancePending,
                              textAlign: TextAlign.center,
                              style: Theme.of(context)
                                  .textTheme
                                  .bodyMedium
                                  ?.copyWith(color: Colors.black54),
                            ),
                          ],
                        ),
                      ),
                    )
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                      itemCount: _entries.length + (_busy ? 1 : 0),
                      itemBuilder: (BuildContext context, int index) {
                        if (index == _entries.length) {
                          return const Align(
                            alignment: Alignment.centerLeft,
                            child: Padding(
                              padding: EdgeInsets.all(12),
                              child: SizedBox(
                                width: 20,
                                height: 20,
                                child:
                                    CircularProgressIndicator(strokeWidth: 2),
                              ),
                            ),
                          );
                        }
                        return _ConversationBubble(entry: _entries[index]);
                      },
                    ),
            ),

            // -- Quick questions -----------------------------------------
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    for (int i = 0; i < 3; i++)
                      Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: ActionChip(
                          label: Text(switch (i) {
                            0 => s.questionSymptoms,
                            1 => s.questionManage,
                            _ => s.questionPrevent,
                          }),
                          onPressed: () => _ask(switch (i) {
                            0 => s.questionSymptoms,
                            1 => s.questionManage,
                            _ => s.questionPrevent,
                          }),
                        ),
                      ),
                  ],
                ),
              ),
            ),

            // -- Input row -------------------------------------------------
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      textInputAction: TextInputAction.send,
                      onSubmitted: _ask,
                      decoration: InputDecoration(
                        hintText: s.yourQuestion,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 14,
                          vertical: 14,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  FilledButton(
                    style: FilledButton.styleFrom(
                      minimumSize: const Size(56, 52),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    onPressed:
                        _busy ? null : () => _ask(_controller.text),
                    child: _busy
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : Icon(Icons.send, size: 20),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ConversationEntry {
  _ConversationEntry.question(this.text)
      : answer = null,
        isQuestion = true;

  _ConversationEntry.answer(AssistantAnswer a)
      : text = a.text,
        answer = a,
        isQuestion = false;

  final String text;
  final AssistantAnswer? answer;
  final bool isQuestion;
}

class _ConversationBubble extends StatelessWidget {
  const _ConversationBubble({required this.entry});

  final _ConversationEntry entry;

  @override
  Widget build(BuildContext context) {
    final bool fromUser = entry.isQuestion;
    final bool isPlaceholder = entry.answer?.isPlaceholder ?? false;

    final Widget bubble = Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.78,
      ),
      decoration: BoxDecoration(
        color: fromUser ? AgriColors.primary : Colors.white,
        borderRadius: BorderRadius.only(
          topLeft: const Radius.circular(14),
          topRight: const Radius.circular(14),
          bottomLeft: Radius.circular(fromUser ? 14 : 4),
          bottomRight: Radius.circular(fromUser ? 4 : 14),
        ),
        border: fromUser ? null : Border.all(color: Colors.green.shade100),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            entry.text,
            style: TextStyle(
              color: fromUser ? Colors.white : Colors.black87,
              fontSize: 14,
              height: 1.4,
            ),
          ),
          if (isPlaceholder) ...[
            const SizedBox(height: 6),
            Icon(Icons.info_outline, size: 14, color: Colors.green.shade400),
          ],
        ],
      ),
    );

    return Align(
      alignment: fromUser ? Alignment.centerRight : Alignment.centerLeft,
      child: bubble,
    );
  }
}
