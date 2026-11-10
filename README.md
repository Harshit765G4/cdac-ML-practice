# cdac_ML_21_oct_2026
ML Training


https://chatgpt.com/share/6af2ad0d-b588-83e8-b89f-9b53306f7f08




name: Main Config
version: 1.0.0
schema: v1

models:
  - name: Gemini 3 Pro Preview
    provider: gemini
    model: gemini-3-pro-preview
    apiKey: ${{ secrets.GEMINI_API_KEY }}
    roles:
      - chat
      - edit
      - apply
    defaultCompletionOptions:
      contextLength: 1048576
      maxTokens: 65536

  - name: Gemini 3 Flash Preview
    provider: gemini
    model: gemini-3-flash-preview
    apiKey: ${{ secrets.GEMINI_API_KEY }}
    roles:
      - chat
      - edit
      - apply
    defaultCompletionOptions:
      contextLength: 1048576
      maxTokens: 65536
