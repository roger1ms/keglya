from hakaton_bot import text_transmission
def split_text(text):
    punctuation_marks = ['.', '?', '!']
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char
        if char in punctuation_marks:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences


text = text_transmission
result = split_text(text)
print(result)
def return_text(result):
    print(result)