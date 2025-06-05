from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence
import torch
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='XPhoneBERTを使用してテキストから音素表現を抽出します')
    parser.add_argument('--text', type=str, default="これ は 、 テスト テキスト です .",
                      help='処理するテキスト（単語分割済み）')
    parser.add_argument('--language', type=str, default='jpn',
                      help='言語コード（ISO 639-3）。デフォルトは日本語（jpn）')
    parser.add_argument('--force-cpu', action='store_true',
                      help='GPUが利用可能でも強制的にCPUを使用')
    parser.add_argument('--output-file', type=str,
                      help='特徴量を保存するファイルパス（指定しない場合は保存しない）')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # デバイスの設定
    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Using device: {device}")

    try:
        # 1. XPhoneBERTモデルとそのトークナイザをロード
        print("Loading XPhoneBERT model and tokenizer...")
        xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to(device)
        tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
        print("Model and tokenizer loaded.")

        # 2. Text2PhonemeSequenceをロード
        print(f"Loading Text2PhonemeSequence for {args.language}...")
        text2phone_model = Text2PhonemeSequence(language=args.language, is_cuda=device.type == 'cuda')
        print("Text2PhonemeSequence loaded.")

        # 3. 入力テキストの処理
        print(f"Input text: {args.text}")

        # 4. テキストを音素シーケンスに変換
        print("Converting text to phoneme sequence...")
        input_phonemes = text2phone_model.infer_sentence(args.text)
        print(f"Phoneme sequence: {input_phonemes}")

        # 5. 音素シーケンスをトークナイズ
        print("Tokenizing phoneme sequence...")
        input_ids = tokenizer(input_phonemes, return_tensors="pt").to(device)
        print(f"Input IDs shape: {input_ids['input_ids'].shape}")

        # 6. XPhoneBERTで特徴量抽出
        print("Extracting features with XPhoneBERT...")
        with torch.no_grad():
            features = xphonebert(**input_ids)
        print("Features extracted.")
        print(f"Output features (last hidden state shape): {features.last_hidden_state.shape}")

        # 特徴量の保存（オプション）
        if args.output_file:
            print(f"Saving features to {args.output_file}...")
            torch.save(features.last_hidden_state.cpu(), args.output_file)
            print("Features saved.")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

# features.last_hidden_state に音素ごとの表現が含まれます。
# features.pooler_output にはシーケンス全体の集約表現が含まれます（BERTの場合）。