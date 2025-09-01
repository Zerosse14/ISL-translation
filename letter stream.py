import time

def process_stream(letter_stream):
    word = ""
    words = []
    previous_letter = None
    repeat_count = 0
    max_repeats = 2  # Maximum allowed repeats for a letter

    for letter in letter_stream():
        if letter == previous_letter:
            repeat_count += 1
        else:
            repeat_count = 0

        # Ignore the letter if it's repeated too many times
        if repeat_count >= max_repeats:
            continue

        if letter.isalpha():  # Check if the character is a valid letter
            word += letter
        elif (letter.isspace() or letter == "<space>") and word:  # Custom space token from the model
            words.append(word)
            word = ""  # Reset for the next word
        elif letter == "<end>" and word:  # Custom end token for sentence/word end
            words.append(word)
            break

        previous_letter = letter  # Update the last seen letter

    if word:  # Append the last word if it exists
        words.append(word)
    
    return words

# Simulated function to mimic real-time letter output from the sign language model with duplicates
def sign_language_model_output():
    # Example: the letter "l" appears twice in "hello" due to model uncertainty
    stream = ["h", "e", "l", "l", "l", "o", "<space>", "w", "o", "r", "r", "r", "l", "d", "<end>"]
    for letter in stream:
        yield letter
        time.sleep(0.5)  # Simulate real-time delay

# Processing the stream of letters from the sign language model
result = process_stream(sign_language_model_output)
print(result)
