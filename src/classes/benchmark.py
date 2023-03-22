
# pylint: disable=missing-function-docstring, invalid-name
from time import time, time_ns
from typing import Any, Generator, Tuple, Optional
import pandas as pd

import nltk

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

sns.set_theme()


class CreativityBenchmark:
    """
        This class is used to benchmark the creativity of a text.
    """
    
    plots_folder = 'plots/'
    tags = set(nltk.tag.mapping._UNIVERSAL_TAGS) # type: ignore
    tags_of_interest = set(['NOUN', 'VERB', 'ADJ', 'ADV'])
    tag_to_embed = {tag: i for i, tag in enumerate(tags)}
    embed_to_tag = {i: tag for i, tag in enumerate(tags)}
    stopwords = set(nltk.corpus.stopwords.words('english'))

    
    def __init__(self, raw_text: str, title: str = "unknown"):
        self.raw_text = raw_text
        self.words = nltk.word_tokenize(raw_text, preserve_line=True)
        self.sents = nltk.sent_tokenize(self.raw_text)
        self.tokenized_sents = [
            nltk.word_tokenize(sent) for sent in self.sents]
        # self.sents = [nltk.word_tokenize(sent) for sent in self.sents]

        # Initialize a list to hold the POS tag counts for each sentence
        self.postag_counts: list[nltk.FreqDist] = []
        self.tagset = ""
        self.title = title

    def ngrams(self, n, **kwargs):
        """Returns ngrams for the text."""
        return nltk.ngrams(self.raw_text, n, kwargs)  # pylint: disable=too-many-function-args

    def sent_postag_counts(self, tagset: str = "universal") -> list[nltk.FreqDist]:
        """Returns sentence-level counts of POS tags for each sentence in the text. """
        if self.postag_counts and self.tagset == tagset:
            return self.postag_counts
        else:
            self.tagset = tagset
            # Collect POS data for each sentence
            for sentence in nltk.pos_tag_sents(self.tokenized_sents, tagset=tagset):
                # Initialize a counter for the POS tags on the sentence level
                lib = nltk.FreqDist()
                for _, token in sentence:
                    lib[token] += 1

                self.postag_counts.append(lib)

            return self.postag_counts

    def book_postag_counts(self, tagset) -> nltk.FreqDist:
        """Get a counter object for the Parts of Speech in the whole book."""

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
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, loc: f"{int(x/1000):,}K"))
        ax1.tick_params(axis='x', labelrotation=90)

        num_tokens_per_sentence = list(self.num_tokens_per_sentence())
        # x = np.arange(0, num_tokens_per_sentence.shape[0])
        # spline = make_interp_spline(x, num_tokens_per_sentence, 3, bc_type='natural')
        ax2.set(title="Distribution of tokens per sentence", xlabel="Sentence #",
                ylabel="(Any) token count", ylim=(10, 300), xlim=(-50, len(num_tokens_per_sentence) + 100))
        ax2.plot(num_tokens_per_sentence)

        fig.subplots_adjust(hspace=0.8)

    def plot_postag_distribution(self, fig=None, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
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
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
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
    def concreteness(word: str, concreteness_df: pd.DataFrame, pos: Optional[str] = None) -> float | None | Any:
        """Returns the mean concreteness rating for a given word, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
        # TODO: Possibly look at amortized values given standard deviations
        try:
            return concreteness_df.loc[word.lower(),"Conc.M"]
        except KeyError:
            return None
        
    def report(self, print_time=True, print_report=True):
        """
            Generates a report for the text.
        """
        if print_time:
            time_now = time()
        ncontent_words = self.ncontent_word_sentlevel()
        avg_num_content_words = sum(ncontent_words)/ len(ncontent_words)
        ratio_content_words = sum(ncontent_words) / len(self.words)
        
        concreteness_df = pd.read_csv('../data/concreteness.txt', sep="\t")
        # concreteness_df.set_index("Word", inplace=True)
        # concreteness_df.sort_index(inplace=True)
        # Fastest way for lookups so far.
        concreteness_df = dict(zip(concreteness_df["Word"], concreteness_df["Conc.M"]))

        # concreteness = list(map(lambda word : self.concreteness(word, concreteness_df), self.words))
        concreteness = list(concreteness_df.get(word.lower(), None) for word in self.words)
        concreteness = [_ for _ in concreteness if _]
        concreteness_num = sum(concreteness) / len(concreteness)
        
        return_string = f"""Text (Title={self.title})
    N words: \t {len(self.words)}
    N sentences: \t {len(self.sents)}
    Average word length: \t {self.avg_word_length():>9.3f}
    Average sentence length: \t {self.avg_sentence_length():>9.3f}
    Average number of tokens per sentence: \t {self.avg_tokens_per_sentence():.3f}
    Average number of content words: \t {avg_num_content_words:.3f} (As proportion: {ratio_content_words:.3%})
    Average concreteness score: \t {concreteness_num:.3f}
    """
    
        if print_time:
            print(f"Report took ~{time() - time_now:.3f}s") # type: ignore
            
        if print_report:
            print(return_string)
        else:
            return return_string


if __name__ == "__main__":
    from nltk.corpus import gutenberg
    
    bench = CreativityBenchmark(gutenberg.raw("austen-emma.txt"), "Emma")
    bench.report()
    
    bench_2 = CreativityBenchmark(gutenberg.raw("bible-kjv.txt"), "Bible")
    bench_2.report()
    
    bench_3 = CreativityBenchmark(gutenberg.raw("carroll-alice.txt"), "Alice in Wonderland")
    bench_3.report()
    # print(report)

    # bench.plot_transition_matrix()
    # plt.show()

    # bench.plot_postag_distribution()
    # plt.show()


    # bench_2 = CreativityBenchmark(gutenberg.raw())
    # print(gutenberg.fileids())