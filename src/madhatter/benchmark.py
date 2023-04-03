"""Main Class for the Application Logic"""

# pylint: disable=missing-function-docstring, invalid-name
from time import time
from typing import Any, Callable, Generator, Optional, Tuple, NamedTuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline
from .utils import get_concreteness_df, mean, slope_coefficient
from .models import sent_predictions

sns.set_theme()

class Report(NamedTuple):
    title: Optional[str] = None
    nwords: Optional[int] = None
    mean_wl: Optional[float] = None
    mean_sl: Optional[float] = None
    mean_tokenspersent: Optional[float] = None
    prop_contentwords: Optional[float] = None
    mean_conc: Optional[float] = None
    prop_pos: Optional[dict] = None


class CreativityBenchmark:
    """
        This class is used to benchmark the creativity of a text.
    """

    plots_folder = 'plots/'
    tags = set(nltk.tag.mapping._UNIVERSAL_TAGS)  # type: ignore
    tags_of_interest = set(['NOUN', 'VERB', 'ADJ'])  # ignore 'ADV'
    tag_to_embed = {tag: i for i, tag in enumerate(tags)}
    embed_to_tag = {i: tag for i, tag in enumerate(tags)}
    stopwords = set(nltk.corpus.stopwords.words('english'))

    def __init__(self, raw_text: str, title: str = "unknown", tagset: str = 'universal'):
        self.raw_text = raw_text
        self.words = nltk.word_tokenize(raw_text, preserve_line=True)
        self.sents = nltk.sent_tokenize(self.raw_text)
        self.tokenized_sents = [
            nltk.word_tokenize(sent) for sent in self.sents]
        
        self.tagset = tagset
        self.tagged_sents = nltk.pos_tag_sents(self.tokenized_sents, tagset=self.tagset)
        # self.sents = [nltk.word_tokenize(sent) for sent in self.sents]

        # Initialize a list to hold the POS tag counts for each sentence
        self.postag_counts: list[nltk.FreqDist] = []
        self.title = title

    def ngrams(self, n, **kwargs):
        """Returns ngrams for the text."""
        return nltk.ngrams(self.raw_text, n, kwargs)  # type: ignore # pylint: disable=too-many-function-args

    def sent_postag_counts(self, tagset: str = "universal") -> list[nltk.FreqDist]:
        """Returns sentence-level counts of POS tags for each sentence in the text. """
        if self.postag_counts and self.tagset == tagset:
            return self.postag_counts
        else:
            self.tagset = tagset
            # Collect POS data for each sentence
            for sentence in self.tagged_sents:
                # Initialize a counter for the POS tags on the sentence level
                lib = nltk.FreqDist()
                for _, token in sentence:
                    lib[token] += 1

                self.postag_counts.append(lib)

            return self.postag_counts

    @property
    def tagged_words(self):
        return nltk.pos_tag(self.words, tagset=self.tagset)

    def book_postag_counts(self, tagset: Optional[str] = None) -> nltk.FreqDist:
        """Get a counter object for the Parts of Speech in the whole book."""

        if not tagset:
            tagset = self.tagset
        # Opt to use this instead for consistency.
        book_total_postags = nltk.FreqDist()
        for l in self.sent_postag_counts(tagset=tagset):
            book_total_postags += l
        return book_total_postags

    def num_tokens_per_sentence(self) -> Generator[int, None, None]:
        """Returns a generator for the number of tokens in each sentence."""
        return (len(sentence) for sentence in self.tokenized_sents)

    def total_tokens_per_sentence(self) -> int:
        return sum(self.num_tokens_per_sentence())

    def avg_tokens_per_sentence(self) -> float:
        return sum(self.num_tokens_per_sentence())/len(self.sents)

    def postag_graph(self):
        # Potentially consider color schemes for nounds, adjectives, etc., not just a random one
        book_total_postags = self.book_postag_counts(tagset='universal')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
        sns.barplot(x=list(book_total_postags.keys()), y=list(
            book_total_postags.values()), label=self.title, ax=ax1)
        ax1.set_title(f"POS Tag Counts for {self.title}")
        ax1.set_ylabel("Count")
        ax1.set_ylim(bottom=30)

        # Set counts to appear with the K suffix
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(  # type: ignore
            lambda x, loc: f"{int(x/1000):,}K"))
        ax1.tick_params(axis='x', labelrotation=90)

        num_tokens_per_sentence = list(self.num_tokens_per_sentence())
        # x = np.arange(0, num_tokens_per_sentence.shape[0])
        # spline = make_interp_spline(x, num_tokens_per_sentence, 3, bc_type='natural')
        ax2.set(title="Distribution of tokens per sentence", xlabel="Sentence #",
                ylabel="(Any) token count", ylim=(10, 300), xlim=(-50, len(num_tokens_per_sentence) + 100))
        ax2.plot(num_tokens_per_sentence)

        fig.subplots_adjust(hspace=0.8)

    def plot_postag_distribution(self, fig=None, ax=None, **kwargs) -> Tuple[Any, Any]:
        '''
        Plots a stackplot of the POS tag counts for each sentence in a book. 
        Note: works best with a Pandas dataframe with the columns as the POS tags and the rows as the sentences.

        TODO: Optionally, set more options for modifying the figure, e.g. linewidth, color palette, etc.
        '''

        df = pd.DataFrame(self.sent_postag_counts(tagset='universal'))
        # Fill in any missing values with 0
        df.fillna(0, inplace=True)
        # Divide each row by the sum of the row to get proportions
        df = df.div(df.sum(axis=1), axis=0)

        if fig is None or ax is None:
            fig, ax = plt.subplots(**kwargs)
        xnew = np.linspace(0, df.shape[0], 100)

        graphs = []
        # For each PosTag, create a smoothed line
        for label in df.columns:
            spl = make_interp_spline(
                list(df.index), df[label], bc_type='natural')  # BSpline object
            power_smooth = spl(xnew)

            graphs.append(power_smooth)

        ax.stackplot(xnew, *graphs, labels=df.columns, linewidth=0.1,
                     colors=sns.color_palette("deep", n_colors=df.shape[1], as_cmap=True))

        ax.set(xbound=(0, df.shape[0]), ylim=(0, 1), title='Parts of Speech in Emma',
               xlabel='Sentence #', ylabel='Proportion of sentence')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(  # type: ignore
            lambda x, loc: f"{x/df.shape[0]:.0%}"))

        ax.legend()
        return fig, ax

    def plot_transition_matrix(self):
        """
        Plots a transition matrix for the POS tags in a book.
        """
        tagged_words = nltk.pos_tag(self.words, tagset='universal')

        counter = np.zeros((len(self.tags), len(self.tags)), dtype=np.int32)
        for pair_1, pair_2 in zip(tagged_words, tagged_words[1:]):
            counter[self.tag_to_embed[pair_1[1]],
                    self.tag_to_embed[pair_2[1]]] += 1
        counter = (counter - counter.mean()) / counter.std()

        plt.figure(figsize=(20, 20))
        plt.imshow(counter, cmap="Blues")
        for i in range(counter.shape[0]):
            for j in range(counter.shape[1]):
                plt.text(j, i, f"{self.embed_to_tag[i]}/{self.embed_to_tag[j]}", ha="center",
                         va="bottom", color="gray", fontdict={'size': 14})
                plt.text(j, i, f"{counter[i, j]:.3f}", ha="center",
                         va="top", color="gray", fontdict={'size': 18})
        plt.title(
            f"POS Tag Transition Matrix for {self.title} with {len(tagged_words)} words")
        plt.axis('off')

    # def get_synsets(word):

    def avg_word_length(self):
        return sum(len(word) for word in self.words)/len(self.words)

    def avg_sentence_length(self):
        return sum(len(sentence) for sentence in self.sents)/len(self.sents)

    def content_words(self):
        return (word for word in self.words if word not in self.stopwords)

    def content_word_sentlevel(self):
        """Discards content words for words 

        Returns:
            list[list[str]]: A list of sentences containing the word tokens.
        """
        return [[word for word in sent if word not in self.stopwords] for sent in self.tokenized_sents]

    def ncontent_word_sentlevel(self):
        return [len(sent) for sent in self.content_word_sentlevel()]

    @staticmethod
    def concreteness(data: str | list[str], concreteness_df: pd.DataFrame) -> float | None | list[float | None]:
        """Returns the mean concreteness rating for a given word or list of words, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
        # TODO: Possibly look at amortized values given standard deviations

        # Fastest way for lookups so far.
        concreteness = dict(
            zip(concreteness_df["Word"], concreteness_df["Conc.M"]))

        if isinstance(data, str):
            return concreteness.get(data.lower(), None)
        if isinstance(data, list):
            return [concreteness.get(w.lower(), None) for w in data if w not in CreativityBenchmark.stopwords]
        raise TypeError(
            f"Inappropriate argument type for `word`. Expected `list` or `str`, but got {type(data)}")

    def calculate_sent_slopes(self, model, n) -> list[list[float]]:
        # Returns slopes for the __words__ of the first `n` sentences of the `sents` list of sentences.
        res = []
        for sent in self.tokenized_sents[:n]:
            results = sent_predictions(sent, self, model)

            res.append(
                [slope_coefficient(
                    np.arange(len(result)),
                    result)
                 for result in results
                 if len(result) > 0]
            )

        return res

    @property
    def model(self):
        return self.model

    @property
    def word2vec_model(self):
        return self.word2vec_model

    def calculate_sim_scores(self, model, sim_function: Callable, max_sents=-1):
        similarity_scores = []
        for sent in self.tokenized_sents[:max_sents]:

            probs, predictions = sent_predictions(
                sent, self, model, return_tokens=True, k=10)
            average_position_of_correct_prediction = 0
            # number of predictions which do not include the true value in the topmost k results
            missed_predictions = 0
            # note that word here is a tuple of the word and its POS tag
            i = 0
            for (word, predlist) in predictions.items():
                try:
                    # print(word[0], predlist)
                    average_position_of_correct_prediction += predlist.index(
                        word[0])
                    i += 1
                except ValueError:
                    missed_predictions += 1

            # Avoid division by zero error
            if i == 0:
                average_position_of_correct_prediction = None
            else:
                average_position_of_correct_prediction /= i
            similarity_scores.append(
                (average_position_of_correct_prediction, missed_predictions))
            break
            #     for item in predlist:
            # similarity_scores.append(
            #     [[sim_function(word[0], pred) for pred in predlist] for word,predlist in predictions.items()]
            # )

        return similarity_scores

    # def sim_func(word: str, pred: str) -> float | None:
    #     """Arbitrary function to use when calculating vector similarity between the embeddings of two words. Serves as an example.

    #     Parameters
    #     ----------
    #     word : str
    #         Normally, the original (true) value.
    #     pred : str
    #         Normally, the predicted value.

    #     Returns
    #     -------
    #     Optional[float]
    #         Can return a float or None.
    #     """
    #     try:
    #         return word2vec_model.similarity(word, pred)
    #     except:
    #         pass

    def report(self, print_time=True, postag_distribution=False) -> Report:
        """
            Generates a report for the text.
        """

        postag_dist = {}
        concreteness_num = 0.0

        if print_time:
            time_now = time()

        ncontent_words = self.ncontent_word_sentlevel()
        # avg_num_content_words = mean(ncontent_words)
        ratio_content_words = sum(ncontent_words) / len(self.words)

        conc_df = get_concreteness_df()

        concreteness = [_ for _ in self.concreteness(
            self.words, conc_df) if _]  # type: ignore
        concreteness_num = mean(concreteness)

        if postag_distribution:

            # The postagging takes a while
            postag_counts = self.book_postag_counts()
            total = sum(i for i in postag_counts.values())
            postag_dist = {"OTHER": 0.0}
            for tag, val in postag_counts.items():
                if tag in self.tags_of_interest:
                    postag_dist[tag] = val/total
                else:
                    postag_dist["OTHER"] += val / total


        result = Report(self.title, len(self.words), self.avg_word_length(), self.avg_sentence_length(), self.avg_tokens_per_sentence(), ratio_content_words, concreteness_num, postag_dist)

        if print_time:
            print(f"Report took ~{time() - time_now:.3f}s")  # type: ignore

        return result
