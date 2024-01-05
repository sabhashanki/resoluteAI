from ai4bharat.transliteration import XlitEngine
e = XlitEngine("ta", beam_width=10, rescore=True)
out = e.translit_word("Good morning", topk=5)
print(out)
