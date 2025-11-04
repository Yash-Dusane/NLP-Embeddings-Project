
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")

def check_ingredient_usage(recipe_text, ingredient):
    doc = nlp(recipe_text)
    ingredient_doc = nlp(ingredient)
    ingredient_lemmas = [token.lemma_.lower() for token in ingredient_doc]
    ingredient_length = len(ingredient_lemmas)

    # Configure substitution patterns
    substitution_patterns = [
        {'pattern': [{'LOWER': 'instead'}, {'LOWER': 'of'}], 'replaced_pos': 'after'},
        {'pattern': [{'LOWER': 'replace'}, {'LOWER': 'with'}], 'replaced_pos': 'before'},
        {'pattern': [{'LOWER': 'substitute'}, {'LOWER': 'with'}], 'replaced_pos': 'before'},
        {'pattern': [{'LOWER': 'replaced'}, {'LOWER': 'with'}], 'replaced_pos': 'before'},
        {'pattern': [{'LOWER': 'swap'}, {'LOWER': 'for'}], 'replaced_pos': 'after'},
    ]
    
    matcher = Matcher(nlp.vocab)
    for i, pattern in enumerate(substitution_patterns):
        matcher.add(f"SUBSTITUTION_{i}", [pattern['pattern']])

    used = False
    for sent in doc.sents:
        sent_doc = nlp(sent.text)
        sentence_lemmas = [token.lemma_.lower() for token in sent_doc]

        # Check for ingredient presence
        for i in range(len(sentence_lemmas) - ingredient_length + 1):
            if sentence_lemmas[i:i+ingredient_length] == ingredient_lemmas:
                start_idx = i
                end_idx = i + ingredient_length
                ingredient_span = sent_doc[start_idx:end_idx]

                # Check for negation
                negated = any(is_negated(token) for token in ingredient_span)

                # Check for substitution
                substituted = False
                matches = matcher(sent_doc)
                for match_id, start, end in matches:
                    pattern = substitution_patterns[match_id]
                    replaced_pos = pattern['replaced_pos']
                    
                    if replaced_pos == 'after':
                        replaced_part = sent_doc[end:]
                    else:
                        replaced_part = sent_doc[:start]

                    if (ingredient_span.start_char >= replaced_part.start_char and
                        ingredient_span.end_char <= replaced_part.end_char):
                        substituted = True
                        break

                if not negated and not substituted:
                    used = True

    return used

def is_negated(token):
    # Check direct negation
    if any(child.dep_ == 'neg' for child in token.head.children):
        return True
    
    # Check ancestor negation
    for ancestor in token.ancestors:
        if any(child.dep_ == 'neg' for child in ancestor.children):
            return True
    
    # Check nearby negation words
    negation_words = {'no', 'not', "n't", 'never', 'without', 'none', 'nobody'}
    for child in token.head.children:
        if child.lower_ in negation_words:
            return True
    
    return False

# Example usage
recipe_text = """
Since this is a veg recipe, so do not add chicken. 
Add 1 cup sugar, 1 cup water, and mix with flour.
"""

print(check_ingredient_usage(recipe_text, "chicken"))  # False
print(check_ingredient_usage(recipe_text, "cow milk"))  # False
print(check_ingredient_usage(recipe_text, "sugar"))     # True
print(check_ingredient_usage(recipe_text, "eggs"))      # True
print(check_ingredient_usage(recipe_text, "flour"))     # True
