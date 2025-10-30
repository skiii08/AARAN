import torch
from pathlib import Path
import numpy as np

# ==================== CONFIGURATION ====================
# プロジェクトのルートパスに合わせて修正してください (通常は変更不要)
BASE_DIR = Path("/Users/watanabesaki/PycharmProjects/AARAN")
PROCESSED_DIR = BASE_DIR / "data" / "processed"

USER_FEATURES_FILE = PROCESSED_DIR / "user_features.pt"
ENTITY_EMBEDDINGS_FILE = PROCESSED_DIR / "entity_embeddings.pt"


# ==================== INSPECTION FUNCTION ====================
def inspect_pt_file(file_path, name):
    print("=" * 60)
    print(f"[{name} の構造確認] ({file_path.name})")
    print("=" * 60)

    if not file_path.exists():
        print(f"❌ ファイルが見つかりません: {file_path}")
        return

    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"❌ ファイルのロード中にエラーが発生しました: {e}")
        return

    # 全ファイル共通: トップレベルキーの確認
    print(f"✅ トップレベルのキー: {list(data.keys())}")

    # --- USER_FEATURES (ユーザー特徴量) の詳細確認 ---
    if name == 'USER_FEATURES':
        print("\n--- 'features' TENSOR ---")
        if 'features' in data:
            print(f" - 'features' Tensor Shape: {data['features'].shape}")

        print("\n--- 'feature_dims' (次元情報) ---")
        if 'feature_dims' in data and isinstance(data['feature_dims'], dict):
            dims_sum = 0
            # 結合順序の推定のため、全次元を表示
            print(" - 結合順序の推定:")
            for k, v in data['feature_dims'].items():
                print(f"   > {k:20s}: {v}")
                if isinstance(v, int):
                    dims_sum += v

            print(f"\n - feature_dims の合計次元数: {dims_sum}")
            if 'features' in data and data['features'].shape[1] == dims_sum:
                print("   ✅ 'features' の次元数と 'feature_dims' の合計が一致しています。")
                print(
                    "   💡 Person Embeddingは結合されたfeatures配列から、この情報を使ってスライス抽出する必要があります。")
            else:
                print("   ⚠️ 'features' の次元数と 'feature_dims' の合計が一致しません。結合順序の推定が必要です。")
        else:
            print(" - 'feature_dims' が見つからないか、期待される辞書形式ではありません。")

    # --- ENTITY_EMBEDDINGS (エンティティ埋め込み) の詳細確認 ---
    elif name == 'ENTITY_EMBEDDINGS':
        print("\n--- エンティティマップ構造 ---")
        for key in ['actors', 'directors']:
            if key in data and isinstance(data[key], dict):
                print(f" - '{key}' (人物名→埋め込みの辞書) のサイズ: {len(data[key])}")

                # 最初のアイテムの構造を確認
                first_key = next(iter(data[key].keys()), None)
                if first_key:
                    first_value = data[key][first_key]
                    value_type = type(first_value).__name__
                    value_len = len(first_value) if hasattr(first_value, '__len__') else 'N/A'

                    print(f"   > 最初の人物名: '{first_key}'")
                    print(f"   > 値の型: {value_type}, 長さ: {value_len}")
                    if value_len == 300:
                        print("   ✅ 埋め込みベクトルの長さが300Dで正しく見えます。")
                    else:
                        print("   ⚠️ 埋め込みベクトルの長さが300Dではありません。")

            elif key in data:
                print(f" - '{key}' が期待される人物名→埋め込みの辞書形式ではありません。型: {type(data[key]).__name__}")

    print("\n")


# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    print(f"ベースディレクトリ: {BASE_DIR}")
    print(f"対象ディレクトリ: {PROCESSED_DIR}\n")

    inspect_pt_file(USER_FEATURES_FILE, "USER_FEATURES")

    inspect_pt_file(ENTITY_EMBEDDINGS_FILE, "ENTITY_EMBEDDINGS")