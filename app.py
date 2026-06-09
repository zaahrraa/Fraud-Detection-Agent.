import os
from flask import Flask, render_template, request, jsonify
from fraud_model import (
    load_dataset,
    prepare_data,
    balance_data,
    split_data,
    train_models,
    evaluate_models,
    summarize_dataset,
)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-key-change-in-production')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def api_train():
    file = request.files.get('dataset')
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Please upload a credit card dataset CSV file.'}), 400

    try:
        df = load_dataset(file)
        summary = summarize_dataset(df)
        balanced = balance_data(prepare_data(df))
        X_train, X_test, y_train, y_test = split_data(balanced)
        models = train_models(X_train, y_train, X_test, y_test)
        metrics = evaluate_models(models, X_test, y_test)

        return jsonify({
            'success': True,
            'summary': summary,
            'balanced_rows': len(balanced),
            'metrics': metrics,
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
