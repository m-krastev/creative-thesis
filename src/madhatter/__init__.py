"""
    Main class
"""

from .benchmark import *
from .models import *
from .utils import *

from .loaders import load_concreteness, load_freq, load_imageability
from nltk import download

# Hardcode download all NLTK resources just in case TODO: FIX
# download('all')

load_freq()
load_concreteness()
load_imageability()

if __name__ == "__main__":
    from nltk.corpus import gutenberg

    bench = CreativityBenchmark(gutenberg.raw("austen-emma.txt"), "Emma")
    bench.report()

    bench_2 = CreativityBenchmark(gutenberg.raw("bible-kjv.txt"), "Bible")
    bench_2.report()

    bench_3 = CreativityBenchmark(gutenberg.raw(
        "carroll-alice.txt"), "Alice in Wonderland")
    bench_3.report()

    # print(report)

    # bench.plot_transition_matrix()
    # plt.show()

    # bench.plot_postag_distribution()
    # plt.show()

    # bench_2 = CreativityBenchmark(gutenberg.raw())
    # print(gutenberg.fileids())
