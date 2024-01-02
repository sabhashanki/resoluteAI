from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
data = 'hi, How are you ?'

print(transliterate(data, sanscript.HK, sanscript.TAMIL))
# print(transliterate(data, sanscript.ITRANS, sanscript.DEVANAGARI))

# scheme_map = SchemeMap(SCHEMES[sanscript.VELTHUIS], SCHEMES[sanscript.TAMIL])

# print(transliterate(data, scheme_map=scheme_map))
