import argparse

from sounddevice import default


def demo():
    parser = argparse.ArgumentParser(description='Demo for Lip2Speech')

    parser.add_argument('--dataset', dest='dataset', required=False,
                    help='name of dataset, choices: LRW, WILD, AVSpeech, GRID', default='LRW')

    parser.add_argument('--root', dest='dataset_path', required=False,
                        help='root path of dataset', default='Datasets/SAMPLE_LRW')

    parser.add_argument('--model_path', dest='saved_model', required=False,
                        help='path of saved_model', default='savedmodels/lip2speech_final.pth')
    
    parser.add_argument('--encoding', dest='encoding', required=False,
                    help='encoding to use for generating speech, choices:  face, voice', default='voice')

    args = parser.parse_args()

    return args


def evaluate():
    parser = argparse.ArgumentParser(description='evaluation of Lip2Speech')
    
    parser.add_argument('--dataset', dest='dataset', required=True,
                        help='name of dataset, choices: LRW, WILD, AVSpeech, GRID', default='LRW')

    parser.add_argument('--root', dest='dataset_path', required=True,
                        help='root path of dataset')

    parser.add_argument('--model_path', dest='saved_model', required=True,
                        help='path of saved_model')
 
    args = parser.parse_args()

    return args


def train():
    parser = argparse.ArgumentParser(description='Trainer of Lip2Speech')
    
    parser.add_argument('--dataset', dest='dataset', required=True,
                        help='name of dataset, choices: LRW, WILD, AVSpeech, GRID', default='LRW')

    parser.add_argument('--root', dest='dataset_path', required=True,
                        help='root path of dataset')

    parser.add_argument('--finetune_model_path', dest='finetune_model', required=False,
                        help='path of finetune_model', default='')
 
    args = parser.parse_args()

    return args

