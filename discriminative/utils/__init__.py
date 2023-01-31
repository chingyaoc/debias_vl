"""
Miscellaneous utilities + setup
"""
import os
from os.path import join


def initialize_save_paths(args):
    # Save embeddings
    if not os.path.exists('./embeddings/'):
        os.makedirs('./embeddings/')
    args.embeddings_dir = join(f'./embeddings/{args.dataset}')
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
        
        
def initialize_experiment(args):
    args.zeroshot_predict_by = 'text'
    base_model = args.load_base_model.replace('/', '_').replace('-', '_')
    initialize_save_paths(args)

    print(f'-> Dataset:         {args.dataset}') # ({args.config})')
    print(f'---')
