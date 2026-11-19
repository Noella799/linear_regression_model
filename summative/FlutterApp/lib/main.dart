import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Deployed FastAPI service on Render.
const String apiUrl = 'https://pciat-regression-api.onrender.com/predict';

void main() => runApp(const PciatApp());

class PciatApp extends StatelessWidget {
  const PciatApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Child Internet-Use Check',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const PredictionPage(),
    );
  }
}

/// Describes one input variable the model needs.
class InputSpec {
  final String key; // JSON field name sent to the API
  final String label; // label shown above the text field
  final String hint; // allowed values / example
  final String kind; // 'int', 'double' or 'text'

  const InputSpec(this.key, this.label, this.hint, this.kind);
}

// One text field per model input variable (10 in total).
const List<InputSpec> inputs = [
  InputSpec('age', 'Age (years)', '5 – 22, e.g. 12', 'int'),
  InputSpec('sex', 'Sex', '0 = male, 1 = female', 'int'),
  InputSpec('internet_hours_per_day', 'Daily internet use',
      '0 = <1h, 1 = ~1h, 2 = ~2h, 3 = 3h or more', 'int'),
  InputSpec('height_in', 'Height (inches)', '30 – 85, e.g. 60', 'double'),
  InputSpec('weight_lb', 'Weight (pounds)', '20 – 350, e.g. 100', 'double'),
  InputSpec('sleep_disturbance_t', 'Sleep disturbance score (SDS T-score)',
      '38 – 100, e.g. 65', 'int'),
  InputSpec('cgas_score', 'General functioning score (CGAS)',
      '1 – 100, e.g. 60', 'int'),
  InputSpec('systolic_bp', 'Systolic blood pressure (mmHg)',
      '60 – 200, e.g. 115', 'double'),
  InputSpec('heart_rate', 'Resting heart rate (bpm)', '40 – 180, e.g. 80',
      'double'),
  InputSpec('enroll_season', 'Enrolment season',
      'Spring, Summer, Fall or Winter', 'text'),
];

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final spec in inputs) spec.key: TextEditingController(),
  };

  bool _loading = false;
  String? _error; // shown in red in the result area
  double? _score;
  String? _riskLevel;
  String? _advice;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  /// Reads every field, checks for missing values and wrong types,
  /// and builds the JSON body. Returns null (and sets _error) if a value is invalid.
  Map<String, dynamic>? _buildBody() {
    final body = <String, dynamic>{};
    for (final spec in inputs) {
      final text = _controllers[spec.key]!.text.trim();
      if (text.isEmpty) {
        _error = 'Please enter a value for "${spec.label}".';
        return null;
      }
      switch (spec.kind) {
        case 'int':
          final value = int.tryParse(text);
          if (value == null) {
            _error = '"${spec.label}" must be a whole number (${spec.hint}).';
            return null;
          }
          body[spec.key] = value;
          break;
        case 'double':
          final value = double.tryParse(text);
          if (value == null) {
            _error = '"${spec.label}" must be a number (${spec.hint}).';
            return null;
          }
          body[spec.key] = value;
          break;
        default:
          // Accept any capitalisation, e.g. "fall" -> "Fall"
          body[spec.key] =
              text[0].toUpperCase() + text.substring(1).toLowerCase();
      }
    }
    return body;
  }

  /// Turns FastAPI's 422 validation response into a readable sentence.
  String _readableValidationError(String responseBody) {
    try {
      final detail = jsonDecode(responseBody)['detail'];
      if (detail is List && detail.isNotEmpty) {
        final first = detail.first;
        final loc = first['loc'] as List;
        final field = loc.isNotEmpty ? loc.last.toString() : '';
        final spec = inputs.where((s) => s.key == field);
        final label = spec.isNotEmpty ? spec.first.label : field;
        return '"$label": ${first['msg']}. Allowed: '
            '${spec.isNotEmpty ? spec.first.hint : 'see the field hint'}.';
      }
    } catch (_) {}
    return 'Some values are out of range. Please check the hints under each field.';
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _error = null;
      _score = null;
    });

    final body = _buildBody();
    if (body == null) {
      setState(() {}); // show the error set in _buildBody
      return;
    }

    setState(() => _loading = true);
    try {
      // A free Render service can take up to a minute to wake up, so allow 90 s.
      final response = await http
          .post(
            Uri.parse(apiUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 90));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _score = (data['predicted_pciat_total'] as num).toDouble();
          _riskLevel = data['risk_level'];
          _advice = data['message'];
        });
      } else if (response.statusCode == 422) {
        setState(() => _error = _readableValidationError(response.body));
      } else {
        setState(() =>
            _error = 'Server error (${response.statusCode}). Please try again.');
      }
    } on TimeoutException {
      setState(() => _error =
          'The server took too long to respond. It may be waking up — try again.');
    } catch (_) {
      setState(() => _error =
          'Could not reach the server. Check your internet connection.');
    } finally {
      setState(() => _loading = false);
    }
  }

  Color _riskColor(String? level) {
    switch (level) {
      case 'High':
        return Colors.red.shade700;
      case 'Moderate':
        return Colors.orange.shade800;
      case 'Mild':
        return Colors.amber.shade800;
      default:
        return Colors.green.shade700;
    }
  }

  Widget _buildResultArea() {
    Widget content;
    if (_loading) {
      content = const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
              width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)),
          SizedBox(width: 12),
          Text('Predicting… (first request may take up to a minute)'),
        ],
      );
    } else if (_error != null) {
      content = Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, color: Colors.red.shade700),
          const SizedBox(width: 8),
          Expanded(
            child: Text(_error!,
                style: TextStyle(color: Colors.red.shade700, fontSize: 15)),
          ),
        ],
      );
    } else if (_score != null) {
      content = Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Predicted PCIAT score (0–100)',
              style: TextStyle(fontSize: 14, color: Colors.black54)),
          const SizedBox(height: 4),
          Text(_score!.toStringAsFixed(1),
              style: const TextStyle(fontSize: 36, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text('Risk level: $_riskLevel',
              style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: _riskColor(_riskLevel))),
          const SizedBox(height: 8),
          Text(_advice ?? '', style: const TextStyle(fontSize: 15)),
        ],
      );
    } else {
      content = const Text(
        'Fill in all fields and tap Predict to see the result here.',
        style: TextStyle(color: Colors.black54),
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _error != null ? Colors.red.shade50 : Colors.teal.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: _error != null ? Colors.red.shade200 : Colors.teal.shade200),
      ),
      child: content,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Child Internet-Use Check'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Enter the child\'s details to estimate their problematic '
                'internet-use score. A score of 40 or more suggests the child '
                'may need support.',
                style: TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 16),
              for (final spec in inputs) ...[
                TextField(
                  controller: _controllers[spec.key],
                  keyboardType: spec.kind == 'text'
                      ? TextInputType.text
                      : TextInputType.numberWithOptions(
                          decimal: spec.kind == 'double'),
                  textInputAction: TextInputAction.next,
                  decoration: InputDecoration(
                    labelText: spec.label,
                    helperText: spec.hint,
                    border: const OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
              ],
              const SizedBox(height: 4),
              SizedBox(
                height: 52,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: const Text('Predict', style: TextStyle(fontSize: 18)),
                ),
              ),
              const SizedBox(height: 16),
              _buildResultArea(),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}
