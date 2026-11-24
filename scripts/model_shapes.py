"""Generate/check layer maps from pinned official model configs; never load weights."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / 'docs/assets/model-specs'
PAGES = {
    'glm-5-1': 'glm-5-1.md',
    'glm-5-2': 'glm-5-2.md',
    'deepseek-v4-flash': 'deepseek-v4.md',
    'deepseek-v4-pro': 'deepseek-v4.md',
    'deepseek-v4-1-flash': 'deepseek-v4-1.md',
}
TITLES = {
    'glm-5-1': 'GLM-5.1',
    'glm-5-2': 'GLM-5.2',
    'deepseek-v4-flash': 'DeepSeek-V4-Flash',
    'deepseek-v4-pro': 'DeepSeek-V4-Pro',
    'deepseek-v4-1-flash': 'DeepSeek-V4.1-Flash',
}


def load_configs():
    manifest = json.loads((SPECS / 'sources.json').read_text(encoding='utf-8'))
    configs = {}
    for slug, source in manifest['models'].items():
        raw = (SPECS / source['config_file']).read_bytes()
        if sha256(raw).hexdigest() != source['sha256']:
            raise ValueError(f'{slug}: config differs from recorded upstream bytes')
        if len(source['revision']) != 40:
            raise ValueError(f'{slug}: expected an immutable commit')
        configs[slug] = json.loads(raw)
    if configs['deepseek-v4-1-flash'] != configs['deepseek-v4-1-flash-1017']:
        raise ValueError('V4.1/1017 configs diverged: revise the comparison prose')
    return configs


def glm_rows(config):
    count = config['num_hidden_layers']
    width = config['hidden_size']
    assert config['qk_nope_head_dim'] + config['qk_rope_head_dim'] == config['qk_head_dim']
    mlps = config.get('mlp_layer_types') or [
        'dense' if i < config['first_k_dense_replace'] else 'sparse' for i in range(count)
    ]
    modes = config.get('indexer_types')
    if modes is None:
        freq = max(config.get('index_topk_freq', 1), 1)
        offset = config.get('index_skip_topk_offset', 2)
        modes = ['full' if max(i-offset+1, 0) % freq == 0 else 'shared' for i in range(count)]
    assert len(mlps) == len(modes) == count
    owner = None
    rows = []
    for i, (mlp, mode) in enumerate(zip(mlps, modes, strict=True)):
        if mode == 'full':
            owner = i
        assert mode in ('full', 'shared') and owner is not None
        ffn = f"Dense F={config['intermediate_size']}" if mlp == 'dense' else (
            f"MoE E={config['n_routed_experts']}, k={config['num_experts_per_tok']}, "
            f"F={config['moe_intermediate_size']}"
        )
        rows.append([str(i), f'`[B,S,{width}]`', ffn, f'{mode}; L{owner}',
                     f"own L{i}: `[B,T,{config['kv_lora_rank']}]` + `[B,T,{config['qk_rope_head_dim']}]`"])
    return rows


def deepseek_rows(raw):
    is_v41 = 'text_config' in raw
    config = raw.get('text_config', raw)
    count, width = config['num_hidden_layers'], config['hidden_size']
    head_dim, streams = config['head_dim'], config['hc_mult']
    ratios = config['compress_ratios']
    assert len(ratios) == count + config['num_nextn_predict_layers']
    assert config['num_attention_heads'] % config['o_groups'] == 0
    assert config['num_attention_heads'] * config['head_dim'] % config['o_groups'] == 0
    kv_owner = index_owner = None
    rows = []
    for i in range(count):
        ratio = ratios[i]
        if is_v41:
            if ratio == 0:
                attention, kv, index = 'SWA', '—', '—'
            else:
                if i in config['kv_source_layer_ids']:
                    kv_owner = i
                if i in config['index_source_layer_ids']:
                    index_owner = i
                assert kv_owner is not None and index_owner is not None
                assert ratios[kv_owner] == ratio
                mode = ('Full' if i in config['kv_source_layer_ids'] else
                        'Reindex' if i in config['index_source_layer_ids'] else 'Reuse')
                attention = f'CSA2 {mode}, r={ratio}'
                positions = 'T' if ratio == 1 else f'floor(T/{ratio})'
                kv = f'L{kv_owner}: `[B,{positions},{head_dim}]`'
                index = f'L{index_owner}; K from L{kv_owner}'
            ffn = 'MoE' + (' + Engram' if i in config['engram_layer_ids'] else '')
        else:
            assert ratio in (0, 4, 128)
            attention = 'SWA' if not ratio else f'SWA + {"CSA" if ratio == 4 else "HCA"}, r={ratio}'
            kv = '—' if not ratio else f'own L{i}: `[B,floor(T/{ratio}),{head_dim}]`'
            index = f'own L{i}' if ratio == 4 else '—'
            ffn = 'Hash-MoE' if i < config['num_hash_layers'] else 'MoE'
        rows.append([str(i), f'`[B,S,{streams},{width}]`', attention, ffn, kv, index])
    return rows


def block(slug, config, lang):
    glm = slug.startswith('glm')
    rows = glm_rows(config) if glm else deepseek_rows(config)
    title = TITLES[slug]
    title += f' · {len(rows)} ' + ('主干层' if lang == 'zh' else 'backbone layers')
    if glm:
        headers = (['层', '层间输入/输出', 'FFN', 'Indexer / 索引来源', '主 KV（每层自有）'] if lang == 'zh'
                   else ['Layer', 'Hidden in/out', 'FFN', 'Indexer / selection source', 'Main KV (owned per layer)'])
    else:
        headers = (['层', '四流残差输入/输出', 'Attention', 'FFN', '全局 KV 来源 / 有效长度', 'Indexer 来源'] if lang == 'zh'
                   else ['Layer', 'Four-stream residual in/out', 'Attention', 'FFN', 'Global KV owner / valid length', 'Indexer source'])
    lines = [f'??? info "{title}"', '', '    | ' + ' | '.join(headers) + ' |',
             '    | ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('    | ' + ' | '.join(row) + ' |' for row in rows)
    return '\n' + '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true', help='Regenerate marked layer tables')
    mode.add_argument('--check', action='store_true', help='Check snapshots and generated tables (default)')
    args = parser.parse_args()
    configs = load_configs()
    mismatches = []
    for lang in ('en', 'zh'):
        for slug, filename in PAGES.items():
            page = ROOT / 'docs' / lang / 'ml/llm/model-structures' / filename
            raw = page.read_text(encoding='utf-8')
            begin = f'<!-- BEGIN GENERATED LAYERS {slug} -->'
            end = f'<!-- END GENERATED LAYERS {slug} -->'
            if raw.count(begin) != 1 or raw.count(end) != 1:
                raise ValueError(f'{page}: missing or duplicate generation markers')
            before, remainder = raw.split(begin)
            current, after = remainder.split(end)
            expected = block(slug, configs[slug], lang)
            if current != expected:
                if args.write:
                    page.write_text(before + begin + expected + end + after, encoding='utf-8')
                else:
                    mismatches.append(str(page.relative_to(ROOT)))
    # Independent arithmetic checks for the documented cache and module examples.
    assert 78 * 4096 * (512 + 64) * 2 == 351 * 1024**2
    assert (512//2 + 512//16) * 5//2 + (128//2 + 128//32) * 5//2 == 890
    assert 40 * 128 * 512 * 2 == 5 * 1024**2
    if mismatches:
        raise SystemExit('Outdated layer tables; run --write:\n' + '\n'.join(mismatches))
    print('PASS: 6 pinned configs; 300 backbone layers in each language; shape/cache arithmetic.')


if __name__ == '__main__':
    main()
