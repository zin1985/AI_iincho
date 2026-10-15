# 4号実LLM発話・内省 比較実験の入口

前段: [60日規則代理実験](../results/9plus3-fourth-speech-reflection-60day.md)。

この段階では**実LLM呼び出し実績0件**。専用モデルの接続先が未設定のため、実LLM発話と主体的内省の改善はまだ判定していない。追加したのは外部LLM接続用の隔離試験入口と独立機械審査で、本番NのDB・人格・目標・記憶へ書き込まない。

## 実行

```sh
python -m unittest discover -s experiments -p 'test_fourth_llm_speech_audit.py'
# 保存済みの実LLM回答を再審査（ネット接続なし）
python experiments/fourth_llm_speech_audit.py --input results/fourth-speech-reflection-samples.json --answers /path/to/model_answers.json --output /tmp/fourth_audit.json
# 実LLMを使う場合、専用のHTTPS/OpenAI互換 chat/completions URLとモデル・API鍵を環境変数で設定
FOURTH_LLM_CHAT_URL=https://your-provider.example/v1/chat/completions
FOURTH_LLM_MODEL=your-model
FOURTH_LLM_API_KEY=your-key
python experiments/fourth_llm_speech_audit.py --input results/fourth-speech-reflection-samples.json --generate --output /tmp/fourth_live_audit.json
```

## 判定上の注意

- 出典IDはモデルへ許可された一覧のみ渡す。代理発話の文面は渡さない。
- 出典ID/JSON形式/沈黙選択/注目対象の逸脱を機械審査する。
- **JSON審査に通っただけでは事実整合性・人格らしさ・独立した意思決定の証明にならない。** 原文のDB照合と人手・別モデルでの盲検評価が次に必要。
- 9＋3と3分類の同一日・同一seedで自然さ、正確性、話題の多様性、沈黙率、繰り返し、自己解釈を比較し、実LLMの出力を保存する。提案行動を自動で実行しない。
- API料金が発生しうるので、呼び出しは --generate 明示時のみ。ローカルOllamaは使用禁止。
