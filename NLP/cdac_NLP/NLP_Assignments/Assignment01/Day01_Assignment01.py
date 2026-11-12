
"""
Title: Machine Translation Using Python
Provider: MyMemory
Source Language: English
Target Language: Hindi
"""

from deep_translator import MyMemoryTranslator


def machine_translation(text):

    try:
        translator = MyMemoryTranslator(
            source="en-GB",
            target="hi-IN"
        )

        translated_text = translator.translate(text)

        return translated_text

    except Exception as e:
        return f"Translation Error: {e}"


# Main Program
print("=" * 50)
print("       MACHINE TRANSLATION SYSTEM")
print("=" * 50)

text = input("\nEnter English text: ")

result = machine_translation(text)

print("\nOriginal English Text:")
print(text)

print("\nTranslated Hindi Text:")
print(result)

print("=" * 50)
