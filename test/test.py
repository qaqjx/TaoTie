import  re

SPECIAL_TOKENS = "[##TAOTIE##]"

def find_special_tokens(prompt):
    matches = [(m.start(), m.end()) for m in re.finditer(re.escape(SPECIAL_TOKENS), prompt)]
    
    result = []
    prev_end = 0
    for start, end in matches:
        result.append(prompt[prev_end:start])
        prev_end = end

    if prev_end < len(prompt):
        result.append(prompt[prev_end:])

    indices = [value[0] - idx * len(SPECIAL_TOKENS) for idx, value in enumerate(matches)]
    
    prompt = ''.join(result)
    if prompt == "":
        indices = []  
    indices = sorted(set(indices))  # Ensure indices are unique and sorted
    return prompt, indices

def test_find_special_tokens():
    text = "abc[##TAOTIE##]def[##TAOTIE##]ghi[##TAOTIE##]"
    a , b = find_special_tokens(text)
    assert a == "abcdefghi"
    assert b == [3, 6, 9]

def test_find_special_tokens_2():
    text = "hello"
    a , b = find_special_tokens(text)
    assert a == "hello"
    assert b == []

def test_find_special_tokens_3():
    text = "[##TAOTIE##]hello"
    a , b = find_special_tokens(text)
    assert a == "hello"
    assert b == [0]

def test_find_special_tokens_4():
    text = "hello[##TAOTIE##]"
    a , b = find_special_tokens(text)
    assert a == "hello"
    assert b == [5]

def test_find_special_tokens_5():
    text = "[##TAOTIE##][##TAOTIE##]"
    a , b = find_special_tokens(text)
    assert a == ""
    assert b == []

def test_random_string():
    import random
    import string
    characters = string.ascii_letters + string.digits
    
    for i in range(200):
      text = ''.join(random.choices(characters, k=i))
      indices = [random.randint(0, i) for _ in range(i)]
      textx = ""
      indices.sort(reverse=True)
      for idx in indices:
        textx = text[:idx] + SPECIAL_TOKENS + text[idx:]
      a, b = find_special_tokens(textx)
      assert a == text
      assert b.sort(reverse=False) == indices.sort(reverse=False)
