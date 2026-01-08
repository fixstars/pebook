# --- 定数定義 ---
# モデルパラメータ (例: Llama3-8B)
MODEL_PARAMS_B = 8 * 10**9  # 80億パラメータ
MODEL_TYPE_BYTES = 2  # FP16 = 2 Bytes

# H100 GPU スペック
H100_PEAK_MEM_BW_TB_S = 3.35  # ピークメモリ帯域幅 (TB/s = 1000^4 B/s)
H100_PEAK_MEM_BW_B_S = H100_PEAK_MEM_BW_TB_S * (1000**4)  # (B/s 単位に変換)
H100_PEAK_FLOPS_T = 989.4  # FP16 Tensorコア (TFLOPS = 10^12 FLOPS)
H100_PEAK_FLOPS = H100_PEAK_FLOPS_T * 10**12  # (FLOPS 単位に変換)

# KVキャッシュ計算用パラメータ (Llama3-8B の値を流用)
KV_CACHE_TYPE_BYTES = 2  # KVキャッシュのデータ型 (FP16 = 2 Bytesと仮定)
NUM_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

from calculate_dataset_result_tokens import get_tokens
import statistics

# --- ヘルパー関数 ---


def bytes_to_tb(byte_value):
    """バイトをテラバイト (TB = 1000^4 B) に変換"""
    return byte_value / (1000**4)


def calculate_kv_cache_bytes(
    batch_size, seq_len, layers, kv_heads, head_dim, type_bytes
):
    """KVキャッシュのサイズをバイト単位で計算"""
    # KキャッシュとVキャッシュのペアなので *2 する
    return int(batch_size * seq_len * layers * kv_heads * head_dim * type_bytes * 2)


def calculate_mbu(
    model_params,
    model_type_bytes,
    kv_cache_bytes,
    observed_throughput_tps,
    peak_mem_bw_tb_s,
):
    """観測されたスループットからMBU (%) を計算"""
    model_bytes = model_params * model_type_bytes
    total_bytes_per_token = model_bytes + kv_cache_bytes

    # 達成されたメモリ帯域幅 (B/s) = 1トークンあたりに必要なバイト数 * スループット
    achieved_mem_bw_b_s = total_bytes_per_token * observed_throughput_tps
    # TB/s に変換
    achieved_mem_bw_tb_s = bytes_to_tb(achieved_mem_bw_b_s)

    # MBU = 達成帯域 / ピーク帯域
    mbu_percentage = (achieved_mem_bw_tb_s / peak_mem_bw_tb_s) * 100
    return mbu_percentage, achieved_mem_bw_tb_s


def calculate_max_throughput_memory_bound(
    model_params,
    model_type_bytes,
    batch_size,
    seq_len,
    layers,
    kv_heads,
    head_dim,
    kv_cache_type_bytes,
    peak_mem_bw_b_s,
):
    """MBU=100% (メモリ律速) の場合の理論最大スループット (tokens/sec) を計算"""
    model_bytes = model_params * model_type_bytes
    kv_cache_bytes = calculate_kv_cache_bytes(
        batch_size, seq_len, layers, kv_heads, head_dim, kv_cache_type_bytes
    )
    # 1ステップ (トークン生成) あたりに移動する総バイト数
    total_bytes_per_step = model_bytes + kv_cache_bytes

    # スループット(tokens/sec) = (ピーク帯域(B/s) * バッチサイズ) / (1ステップあたりの総バイト数(B))
    max_throughput = (peak_mem_bw_b_s * batch_size) / total_bytes_per_step
    return max_throughput


def calculate_max_throughput_compute_bound(model_params, peak_flops):
    """MFU=100% (計算律速) の場合の理論最大スループット (tokens/sec) を計算"""
    # 1トークンあたりのFLOPs ≒ 2 * モデルパラメータ数
    flops_per_token = 2 * model_params

    # スループット(tokens/sec) = ピークFLOPs / 1トークンあたりFLOPs
    max_throughput = peak_flops / flops_per_token
    return max_throughput


# --- メイン処理 ---
if __name__ == "__main__":
    # --- MBU 計算例 (1GPUの場合) ---
    print("--- MBU 計算例 (1GPU, Batch=1) ---")
    example_batch_size = 1
    example_observed_throughput = 47  # tokens/sec (実測値の例として残しておく)

    # シーケンス長計算用パラメータ
    input_tokens_list, output_tokens_list = get_tokens()
    avg_input_tokens = sum(input_tokens_list) / len(input_tokens_list)
    avg_new_tokens = sum(output_tokens_list) / len(output_tokens_list)
    # 概算シーケンス長 = 平均入力長 + 生成トークン長の平均値
    approx_seq_len = avg_input_tokens + avg_new_tokens / 2

    kv_bytes_example = calculate_kv_cache_bytes(
        example_batch_size,
        approx_seq_len,
        NUM_LAYERS,
        NUM_KV_HEADS,
        HEAD_DIM,
        KV_CACHE_TYPE_BYTES,
    )

    mbu, achieved_bw = calculate_mbu(
        MODEL_PARAMS_B,
        MODEL_TYPE_BYTES,
        kv_bytes_example,
        example_observed_throughput,
        H100_PEAK_MEM_BW_TB_S,
    )
    print(f"モデル: {MODEL_PARAMS_B / 1e9:.1f}B パラメータ (FP16)")
    print(f"平均入力長: {avg_input_tokens}")
    print(f"生成トークン長の平均値: {avg_new_tokens}")
    print(f"概算シーケンス長: {approx_seq_len}")
    print(
        f"KVキャッシュサイズ (Batch={example_batch_size}): {bytes_to_tb(kv_bytes_example) * 1000**4 / 1024**3:.2f} GB"
    )  # GB表示用に1024ベースに変換
    print(f"観測スループット例: {example_observed_throughput} tokens/sec")
    print(f"達成メモリ帯域: {achieved_bw:.2f} TB/s")
    print(f"H100 ピークメモリ帯域: {H100_PEAK_MEM_BW_TB_S:.2f} TB/s")
    print(f"計算された MBU: {mbu:.2f} %\n")

    # --- 理論最大スループット計算 (メモリ律速 vs 計算律速) ---
    print("--- 理論最大スループット (メモリ律速 vs 計算律速) ---")

    compute_bound_throughput = calculate_max_throughput_compute_bound(
        MODEL_PARAMS_B, H100_PEAK_FLOPS
    )

    print(
        f"計算律速時の理論最大スループット (MFU=100%): {compute_bound_throughput:,.0f} tokens/sec"
    )

    # --- 各サンプルの理論最大スループット (追加機能) ---
    print("\n" + "-" * 60)
    print("各サンプルの理論最大スループット (Batch=1, MBU=100%)")
    print("-" * 60)
    print(f"{'Sample ID':<10} | {'Input':<8} | {'Output':<8} | {'Seq Len':<10} | {'Max Throughput (tokens/sec)':<30}")
    print("-" * 60)

    for i, (in_tok, out_tok) in enumerate(zip(input_tokens_list, output_tokens_list)):
        # 各サンプルの概算シーケンス長 (入力 + 出力/2)
        sample_seq_len = in_tok + out_tok / 2

        throughput_val = calculate_max_throughput_memory_bound(
            MODEL_PARAMS_B,
            MODEL_TYPE_BYTES,
            1,  # Batch size 1 で計算
            sample_seq_len,
            NUM_LAYERS,
            NUM_KV_HEADS,
            HEAD_DIM,
            KV_CACHE_TYPE_BYTES,
            H100_PEAK_MEM_BW_B_S,
        )
        print(f"{i:<10} | {in_tok:<8} | {out_tok:<8} | {sample_seq_len:<10.1f} | {throughput_val:,.0f}")
    print("-" * 60)

    # 理論値の表示 (表形式)
    print("-" * 60)
    print("メモリ律速時の理論最大スループット (MBU=100%)")
    print("-" * 60)
    print(f"{'Batch Size':<12} | {'Max Throughput (tokens/sec)':<30}")
    print("-" * 60)
    display_batches = [
        1,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8092,
        16384,
    ]
    for b in display_batches:
        throughput_val = calculate_max_throughput_memory_bound(
            MODEL_PARAMS_B,
            MODEL_TYPE_BYTES,
            b,
            approx_seq_len,
            NUM_LAYERS,
            NUM_KV_HEADS,
            HEAD_DIM,
            KV_CACHE_TYPE_BYTES,
            H100_PEAK_MEM_BW_B_S,
        )

        throughput_val_samples = []
        for i, (in_tok, out_tok) in enumerate(zip(input_tokens_list, output_tokens_list)):
            # 各サンプルの概算シーケンス長 (入力 + 出力/2)
            sample_seq_len = in_tok + out_tok / 2

            throughput_val_sample = calculate_max_throughput_memory_bound(
                MODEL_PARAMS_B,
                MODEL_TYPE_BYTES,
                b,
                sample_seq_len,
                NUM_LAYERS,
                NUM_KV_HEADS,
                HEAD_DIM,
                KV_CACHE_TYPE_BYTES,
                H100_PEAK_MEM_BW_B_S,
            )
            throughput_val_samples.append(throughput_val_sample)
        print(f"{b:<12} | {throughput_val:,.0f} tokens/sec vs {statistics.harmonic_mean(throughput_val_samples):,.0f} tokens/sec")
    print("-" * 60)
