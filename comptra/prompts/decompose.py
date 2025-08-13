DECOMPOSE = """
We would like to derive a list of short sentences from long and convoluted sentences. For each long sentence, you will use punctuation (e.g., comma, semicolon, etc.), coordinating conjunctions (e.g., for, and, etc.), subordinating conjunctions (e.g., although, because) etc. to divide the sentence into multiple short sentences, which are easy to understand.
Ensure that each of the short sentences reflects a part of the larger sentence.
Here are some examples.

###

Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The Boolean satisfiability problem is a well-researched problem. 
    2. It has many exemplar solvers are available.
    3. It is very fast.
    4. The package solving complexity is very low. 
    5. This is compared to other areas where SAT solvers are used.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Dore was offered several one-off shows in night clubs.
    2. Her best album was rereleased in 2001.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. Jim briefly transfers to the Stamford branch.
    2. Pam confirmed her commitment to Roy.
    3. Corporate is forced to merge the Stamford branch and staff.
    4. The merge is into the Scranton branch.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broke, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. But Jack could not get back to his own time.
    2. One of the drug vials had broke.
    3. There was only enough left in one of the vials.
    4. This was to stop Whistler.

###

Sentence
However, his nonconformist background came to the fore again when he became friendly with William Durning around 1817, having rented a cottage from another member of the Durning family, and on 1 September 1820 he married William's daughter, Emma.

Propositions
    1. However, his nonconformist background came to the fore again.
    2. He became friendly with William Durning around 1817.
    3. He rented a cottage from another member of the Durning family.
    4. He married William's daughter.
    5. The marriage was on 1 September 1820.
    6. William's daughter was Emma.

###

Sentence
Mallzee was founded in December 2012 by Cally Russell and is based in Edinburgh.

Propositions
    1. Mallzee was founded in December 2012.
    2. Mallzee was founded by Cally Russell.
    3. Mallzee is based in Edinburgh.

###

Sentence
He was educated at William Ellis School before being accepted into University College London to study botany and zoology, after graduating he went to the College of the Pharmaceutical Society and studied pharmacy, graduating in 1935. 

Propositions
    1. He was educated at William Ellis School.
    2. This was before being accepted into University College London.
    3. This was to study botany and zoology. 
    4. After graduating he went to the College of the Pharmaceutical Society.
    5. He studied pharmacy.
    6. He graduated in 1935.

###

Sentence
Out of 3 other surrounding neighborhoods, Mattapan saw a population decrease but has the highest proportion of Black/African American residents in the city, but the number of blacks actually dropped over the last decade.

Propositions
    1. Out of 3 other surrounding neighborhoods.
    2. Mattapan saw a population decrease.
    3. Mattapan has the highest proportion of Black/African American residents in the city.
    4. The number of blacks actually dropped over the last decade.

###

Sentence
Nerepis is situated on the Nerepis River and is located east of the town of Grand Bay-Westfield in the Saint John, the nearest city, which is about twenty-five minutes away. 

Propositions
    1. Nerepis is situated on the Nerepis River.
    2. Nerepis is located east of the town of Grand Bay-Westfield.
    3. Grand Bay-Westfield is in the Saint John.
    4. Saint John is the nearest city.
    5. Saint John is about twenty-five minutes from Nerepis.

###

Sentence
In 1961, when Muskee was 20 years old, his mother died, and a year later his grandmother died.

Propositions
    1. In 1961, when Muskee was 20 years old.
    2. His mother died.
    3. A year later, his grandmother died.

###

"""

DECOMPOSE_FRENCH = """
We would like to derive a list of short sentences from long and convoluted sentences. For each long sentence, you will use punctuation (e.g., comma, semicolon, etc.), coordinating conjunctions (e.g., for, and, etc.), subordinating conjunctions (e.g., although, because) etc. to divide the sentence into multiple short sentences, which are easy to understand.
Ensure that each of the short sentences reflects a part of the larger sentence.
Here are some examples.

###

Sentence
Le problème de la satisfaction booléenne est un problème bien étudié avec de nombreux exemples de solveurs disponibles ; il est très rapide, car la complexité de résolution du package est très faible par rapport à d'autres domaines où les solveurs SAT sont utilisés.

Propositions
    1. Le problème de la satisfaction booléenne est un problème bien étudié.
    2. Il existe de nombreux exemples de solveurs disponibles.
    3. Il est très rapide.
    4. La complexité de résolution du package est très faible.
    5. Ceci est comparé à d'autres domaines où les solveurs SAT sont utilisés.

###

Sentence
Dore s'est vu proposer plusieurs concerts ponctuels dans des boîtes de nuit, et son meilleur album a été réédité en 2001.

Propositions
    1. Dore s'est vu proposer plusieurs concerts ponctuels dans des boîtes de nuit.
    2. Son meilleur album a été réédité en 2001.

###

Sentence
Jim est brièvement muté à la succursale de Stamford après que Pam a confirmé son engagement envers Roy, avant que l'entreprise ne soit obligée de fusionner la succursale de Stamford et le personnel dans la succursale de Scranton.

Propositions
    1. Jim est brièvement muté à la succursale de Stamford.
    2. Pam a confirmé son engagement envers Roy.
    3. L'entreprise est obligée de fusionner la succursale de Stamford et le personnel.
    4. La fusion se fait dans la succursale de Scranton.

###

Sentence
Mais Jack ne pouvait pas retourner à son époque, car l'une des fioles de drogue s'était cassée, et il ne restait dans l'une des fioles que suffisamment de drogue pour arrêter Whistler.

Propositions
    1. Mais Jack ne pouvait pas retourner à son époque.
    2. L'une des fioles de drogue s'était cassée.
    3. Il ne restait dans l'une des fioles que suffisamment de drogue.
    4. C'était pour arrêter Whistler.

###

Sentence
Cependant, son passé non-conformiste est revenu au premier plan lorsqu'il s'est lié d'amitié avec William Durning vers 1817, après avoir loué un cottage à un autre membre de la famille Durning, et le 1er septembre 1820, il a épousé la fille de William, Emma.

Propositions
    1. Cependant, son passé non-conformiste est revenu au premier plan.
    2. Il s'est lié d'amitié avec William Durning vers 1817.
    3. Il a loué un cottage à un autre membre de la famille Durning.
    4. Il a épousé la fille de William.
    5. Le mariage a eu lieu le 1er septembre 1820.
    6. La fille de William était Emma.

###

Sentence
Mallzee a été fondée en décembre 2012 par Cally Russell et est basée à Édimbourg.

Propositions
    1. Mallzee a été fondée en décembre 2012.
    2. Mallzee a été fondée par Cally Russell.
    3. Mallzee est basée à Édimbourg.

###

Sentence
Il a fait ses études à la William Ellis School avant d'être accepté à l'University College de Londres pour étudier la botanique et la zoologie. Après avoir obtenu son diplôme, il est allé au College of the Pharmaceutical Society et a étudié la pharmacie, obtenant son diplôme en 1935.

Propositions
    1. Il a fait ses études à la William Ellis School.
    2. C'était avant d'être accepté à l'University College de Londres.
    3. C'était pour étudier la botanique et la zoologie.
    4. Après avoir obtenu son diplôme, il est allé au College of the Pharmaceutical Society.
    5. Il a étudié la pharmacie.
    6. Il a obtenu son diplôme en 1935.

###

Sentence
Parmi les trois autres quartiers environnants, Mattapan a connu une baisse de sa population mais compte la plus forte proportion de résidents noirs/afro-américains de la ville, mais le nombre de noirs a en fait diminué au cours de la dernière décennie.

Propositions
    1. Parmi les trois autres quartiers environnants.
    2. Mattapan a connu une baisse de sa population.
    3. Mattapan a la plus forte proportion de résidents noirs/afro-américains de la ville.
    4. Le nombre de noirs a en fait diminué au cours de la dernière décennie.

###

Sentence
Nerepis est située sur la rivière Nerepis et se trouve à l'est de la ville de Grand Bay-Westfield dans la région de Saint John, la ville la plus proche, qui se trouve à environ vingt-cinq minutes.

Propositions
    1. Nerepis est située sur la rivière Nerepis.
    2. Nerepis est située à l'est de la ville de Grand Bay-Westfield.
    3. Grand Bay-Westfield se trouve dans la région de Saint John.
    4. Saint John est la ville la plus proche.
    5. Saint John est à environ vingt-cinq minutes de Nerepis.

###

Sentence
En 1961, lorsque Muskee avait 20 ans, sa mère est décédée, et un an plus tard, sa grand-mère est décédée.

Propositions
    1. En 1961, lorsque Muskee avait 20 ans.
    2. Sa mère est décédée.
    3. Un an plus tard, sa grand-mère est décédée.

###
    
"""

PARAPHRASE = """
We would like to propose a list of paraphrases of sentences. For each sentence, you will provide four paraphrases that have the same meaning as the original sentence and mostly use the same words as well.
Ensure that each of the four paraphrases is a correct sentence and does not change the meaning of the original sentence.
Here are some examples.

###

Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The Boolean satisfiability problem is a widely studied topic, with numerous exemplar solvers available; it is efficient, as solving package complexity is significantly lower than in other domains using SAT solvers.
    2. Boolean satisfiability, a well-researched problem, boasts many exemplar solvers, and its speed is notable due to the low complexity of package solving compared to other SAT applications.
    3. The problem of Boolean satisfiability has been extensively researched, leading to the development of many exemplar solvers; package solving in this context is fast, given its comparatively low complexity in contrast to other SAT solver uses.
    4. With numerous exemplar solvers available, the Boolean satisfiability problem is well-researched and demonstrates remarkable speed, as the complexity of package solving is much lower than in other SAT solver applications.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Dore’s best album was rereleased in 2001, and she was offered several one-off shows in night clubs.
    2. In 2001, Dore’s best album was rereleased, and she received offers for several one-off performances in night clubs.
    3. Several one-off shows in night clubs were offered to Dore, and her best album saw a rerelease in 2001.
    4. Dore was given opportunities for one-off performances in night clubs, and her best album was rereleased during 2001.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. After Pam confirmed her commitment to Roy, Jim briefly transfers to the Stamford branch, only for corporate to merge Stamford staff into the Scranton branch.
    2. Jim transfers briefly to the Stamford branch after Pam confirms her commitment to Roy, but corporate later merges the Stamford staff into the Scranton branch.
    3. Pam's confirmation of her commitment to Roy leads Jim to briefly transfer to the Stamford branch, which is later merged into the Scranton branch by corporate.
    4. Before corporate merges the Stamford branch and its staff into the Scranton branch, Jim briefly transfers there after Pam confirms her commitment to Roy.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broke, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. Jack could not return to his own time because one of the drug vials had broken, leaving only enough in one vial to stop Whistler.
    2. Since one of the drug vials had broken, Jack was unable to get back to his own time, with just enough remaining in a single vial to stop Whistler.
    3. Because one of the vials of the drug had broken, Jack could not make it back to his own time, as only one vial had enough left to stop Whistler.
    4. One of the drug vials had broken, leaving Jack unable to return to his own time, with only enough left in one vial to stop Whistler.

###

"""

PARAPHRASE_3 = """
We would like to propose a list of paraphrases of sentences. For each sentence, you will provide three paraphrases that have the same meaning as the original sentence and mostly use the same words as well.
Ensure that each of the three paraphrases is a correct sentence and does not change the meaning of the original sentence.
Here are some examples.

###

Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The Boolean satisfiability problem is a widely studied topic, with numerous exemplar solvers available; it is efficient, as solving package complexity is significantly lower than in other domains using SAT solvers.
    2. Boolean satisfiability, a well-researched problem, boasts many exemplar solvers, and its speed is notable due to the low complexity of package solving compared to other SAT applications.
    3. The problem of Boolean satisfiability has been extensively researched, leading to the development of many exemplar solvers; package solving in this context is fast, given its comparatively low complexity in contrast to other SAT solver uses.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Dore’s best album was rereleased in 2001, and she was offered several one-off shows in night clubs.
    2. In 2001, Dore’s best album was rereleased, and she received offers for several one-off performances in night clubs.
    3. Several one-off shows in night clubs were offered to Dore, and her best album saw a rerelease in 2001.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. After Pam confirmed her commitment to Roy, Jim briefly transfers to the Stamford branch, only for corporate to merge Stamford staff into the Scranton branch.
    2. Jim transfers briefly to the Stamford branch after Pam confirms her commitment to Roy, but corporate later merges the Stamford staff into the Scranton branch.
    3. Pam's confirmation of her commitment to Roy leads Jim to briefly transfer to the Stamford branch, which is later merged into the Scranton branch by corporate.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broke, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. Jack could not return to his own time because one of the drug vials had broken, leaving only enough in one vial to stop Whistler.
    2. Since one of the drug vials had broken, Jack was unable to get back to his own time, with just enough remaining in a single vial to stop Whistler.
    3. Because one of the vials of the drug had broken, Jack could not make it back to his own time, as only one vial had enough left to stop Whistler.

###

"""


def get_divide_prompt(sentence: str, mode: str = None):
    if not mode or mode in ["vanilla", "English"]:
        prompt = DECOMPOSE.strip() + f"\n\nSentence\n{sentence}\n\nPropositions"
    elif mode == "French":
        prompt = DECOMPOSE_FRENCH.strip() + f"\n\nSentence\n{sentence}\n\nPropositions"
        return prompt
    elif mode == "paraphrase":
        prompt = PARAPHRASE.strip() + f"\n\nSentence\n{sentence}\n\nPropositions"
    elif mode == "paraphase-3":
        prompt = PARAPHRASE_3.strip() + f"\n\nSentence\n{sentence}\n\nPropositions"
    else:
        raise ValueError(f"Unsupported decomposition mode {mode}")
    # prompt = DECOMPOSE.strip() + f"\n\nSentence\n{sentence}\n\nPropositions"
    # prompt = PARAPHRASE.strip() + f"\n\nSentence\n{sentence}\n\nPropositions"
    return prompt


from typing import List
from sentence_splitter import SentenceSplitter


def sentence_split(text, n_splits=None, lang="en") -> List[str]:
    """
    Use SentenceSplitter (https://github.com/mediacloud/sentence-splitter) in order to split an input text into multiple sentences
    Arguments 
    ----------
        - text : str
            Input document or sequence of words.
        - n_splits : int
            Maximum number of split. If None it will be taken as the maximum possible.
        - lang : str
            Language code of the input sequence.
    Returns
    -------
        - List[str]
            List of sentences in `text`.
    """
    assert lang in [
        "ca",
        "cs",
        "da",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "is",
        "it",
        "lv",
        "lt",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "es",
        "sv",
        "tr",
    ], f"Language code {lang} is not supported by `SentenceSplitter`."
    splitter = SentenceSplitter(language=lang)
    sentences = splitter.split(text=text)
    if n_splits == -1 or n_splits is None:
        return sentences
    else:
        assert n_splits > 0, f"The number of split ({n_splits}) should be > 0!"
        # len = n_splits * b + r
        # first r groups will have a size equal to (b + 1), the remaining (q - r) will have a size of b
        b = len(sentences) // n_splits
        r = len(sentences) % n_splits
        left = [
            " ".join(sentences[i : i + b + 1]) for i in range(0, r * (b + 1), b + 1)
        ]
        right = [
            " ".join(sentences[i : i + b])
            for i in range(r * (b + 1), len(sentences), b)
        ]
        return left + right


def equal_split(text: str, n_splits: int):
    """
    Split an input text into multiple equal length propositions. Each part is separated from the adjacent one by space, in order
    to prevent splitting words in the sentence.
    Arguments 
    ----------
        - text : str
            Input document or sequence of words.
        - n_splits : int
            Maximum number of split. If None, no split is done.
    Returns
    -------
        - List[str]
            List of sentences in `text`.
    """
    if n_splits is None:
        return [text]

    b = len(text) // n_splits
    r = len(text) % n_splits
    print(f"b = {b}, r = {r}")
    last_end = None
    sentences = []
    for i in range(0, r * (b + 1), b + 1):
        if last_end is None:
            # consider text[i : i + b + 1]
            start = i
        else:
            start = last_end
        end = min(i + b + 1, len(text))
        if text[end - 1] == " ":
            sentences.append(text[start:end])
            last_end = end
            continue
        # Make sure not to cut in the middle of a word
        index_previous_space = text[0:end].rfind(" ")
        index_next_space = text[end:].find(" ")
        if index_previous_space < 0:
            if index_next_space < 0:
                # Should never happen
                pass
            else:
                end = end + index_next_space
        else:
            if index_next_space < -1:
                end = index_previous_space
            else:
                if end - index_previous_space > index_next_space:
                    end = end + index_next_space
                else:
                    end = index_previous_space
        sentences.append(text[start:end])
        last_end = end

    for i in range(r * (b + 1), len(text), b):
        if last_end is None:
            # consider text[i : i + b]
            start = i
        else:
            start = last_end

        end = min(i + b, len(text))

        if text[end - 1] == " ":
            sentences.append(text[start:end])
            last_end = end
            continue
        # Make sure not to cut in the middle of a word
        index_previous_space = text[0:end].rfind(" ")
        index_next_space = text[end:].find(" ")
        if index_previous_space < 0:
            if index_next_space < 0:
                # Should never happen
                pass
            else:
                end = end + index_next_space
        else:
            if index_next_space < -1:
                end = index_previous_space
            else:
                if end - index_previous_space > index_next_space:
                    end = end + index_next_space
                else:
                    end = index_previous_space
        sentences.append(text[start:end])
        last_end = end

    return [sentence.strip() for sentence in sentences]


def characterwise_split(x, n_splits, char=" ") -> List[str]:
    sentences = x.split(char)
    if n_splits == -1:
        return sentences
    b = len(sentences) // n_splits
    if b == 0:
        return sentences

    r = len(sentences) % n_splits
    left = [" ".join(sentences[i : i + b + 1]) for i in range(0, r * (b + 1), b + 1)]
    right = [
        " ".join(sentences[i : i + b]) for i in range(r * (b + 1), len(sentences), b)
    ]
    return left + right


#import nltk

#nltk.download("punkt")
#nltk.download("punkt_tab")
#nltk.download("stopwords")
#from nltk.corpus import stopwords

#stop_words = set(stopwords.words("english"))
import nltk
import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Download took too long!")

# Set the signal handler and timeout (e.g., 10 seconds)
signal.signal(signal.SIGALRM, handler)
signal.alarm(10)  # Timeout after 10 seconds

try:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    signal.alarm(0)  # Cancel alarm if successful
    stop_words = set(stopwords.words("english"))
except TimeoutException:
    print("Download timed out!")
    stop_words = []
import string


def keyword_splitting(x: str, n_splits: int = -1) -> List[str]:
    """
    Arguments
    ---------
    - x: str
        Sentence to `split` (to extract keywords from)
    - n_splits: int
        Number of keyword to extract from the sentence
    """
    words = nltk.word_tokenize(x)
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    dedup_words = []
    for w in words:
        if w in dedup_words:
            continue
        if all([c.low() not in "abcdefghijklmnopqrstuvwxyz" for c in w]):
            continue
        if len(w.strip()) <= 2:
            continue
        dedup_words.append(w)
    words = dedup_words
    """
    words = list(
        set([word.lower() for word in words if word[0].upper() != word[0]])
    )
    
    words = [word for word in words if len(word) >= 4 and not word.endswith("ed")]
    words = [w + "." if not w.endswith(".") else w for w in words]
    """
    if n_splits is None or n_splits < 0:
        return words
    print(words[:n_splits])
    return words[:n_splits]


import spacy

nlp = spacy.load("en_core_web_sm")


def structural(
    sentence: str, n_splits: int = -1, max_split_length: int = 4
) -> List[str]:
    """
    Take as input a sentence and decompose it into a given number of splits based
    on the structure of the dependency tree.
    Arguments
    ---------
        - sentence: str,
            sentence to decompose into smaller entities.
        - n_splits: str,
            number of smaller entities, if set to -1 it will consider all of them.
        - max_split_length:
            The maximum length of each entity if the split is done to the maximum.
    """
    stack = [sentence]
    subparts = []
    punctuations = [",", ".", '"', "'", "?", "!"]
    while stack:
        if n_splits > 0 and len(stack) + len(subparts) >= n_splits:
            subparts = stack + subparts
            break

        element = stack.pop()

        # print(subparts)
        if len(element.split(" ")) <= max_split_length:
            if len(element.split(" ")) == 1:
                if len(element) <= 5 or element in stop_words:
                    pass
                else:
                    subparts.append(element)
            elif len(element.split(" ")) == 2:
                if all([len(component) <= 3 for component in element.split(" ")]):
                    pass
                else:
                    subparts.append(element)
            else:
                subparts.append(element)
            continue

        doc = nlp(element)
        root = [token for token in doc if token.head == token][0]
        if root.i == 0:  # root at the start of the sentence
            left = doc[root.i].text
            right = doc[root.i + 1 :].text
        elif root.i == len(doc) - 1:  # root at the end of the sentence
            left = doc[: root.i].text
            right = doc[root.i].text
        else:
            left = doc[: root.i + 1].text
            right = doc[root.i + 1 :].text

        while any([left.startswith(col) for col in punctuations]):
            left = left[1:].strip()
        while any([right.startswith(col) for col in punctuations]):
            right = right[1:].strip()

        if left and right:
            # Process the longest part first
            if len(left.split(" ")) >= len(right.split(" ")):
                stack = [right, left] + stack
            else:
                stack = [left, right] + stack
        else:
            if left:
                stack = [left] + stack
            if right:
                stack = [right] + stack

    return sorted(subparts, key=lambda x: sentence.find(x))


if __name__ == "__main__":
    pass
