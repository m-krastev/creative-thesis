import argparse

def main():
    parser = argparse.ArgumentParser(
        prog='madhatter',
        description='A command-line utility for generating book project reports',
    )

    parser.add_argument(
        'filename',
        help="Text file to parse"
    )

    parser.add_argument('-p', '--postag', action="store_true",
                        help='Whether to return a POS tag distribution over the whole text.')

    parser.add_argument('-u', '--usellm', action="store_true",
                        help='Whether to run GPU-intensive LLMs for additional characteristics')

    parser.add_argument(
        '-m', '--maxtokens', help="Maximum number of predicted tokens for the heavyweight metrics. Tokens start from the beginning of text, -1 to read until the end", default=1000, type=int
    )

    parser.add_argument(
        '-c', '--context', help='Context length for sliding window predictions as part of heavyweight metrics.', default=10, type=int
    )

    parser.add_argument(
        '-t', '--title', help='Optional title to use for the report project.'
    )

    parser.add_argument(
        '-d', '--tagset', help='Tagset to use', default="universal"
    )

    args = parser.parse_args()
    
    from .benchmark import CreativityBenchmark

    with open(args.filename) as f:
        text = f.read()

    bench = CreativityBenchmark(text, args.title, args.tagset)

    print(bench.report(False, args.postag,
          args.usellm, n=args.maxtokens, k=args.context))


main()