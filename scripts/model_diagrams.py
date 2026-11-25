"""Draw bilingual, code-native SVG explanations from pinned model configurations."""
import argparse
from html import escape
from pathlib import Path

from model_shapes import load_configs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/model-diagrams'
COLORS = {
    'compute': ('#e9f0ff', '#3970ce'),
    'cache': ('#e5f6f2', '#20816f'),
    'index': ('#fff3d9', '#bc7b12'),
    'expert': ('#f0eafd', '#8051bd'),
    'memory': ('#fceaf2', '#b94f80'),
    'neutral': ('#f0f3f8', '#718198'),
}


class Figure:
    def __init__(self, title, subtitle, height=600):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
                      f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                      'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                      '<path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
                      '<style>text{font-family:Segoe UI,Microsoft YaHei,Noto Sans CJK SC,Arial,sans-serif;'
                      'fill:#20324b} .shape{font-family:Consolas,Microsoft YaHei,monospace}</style>',
                      f'<rect width="1120" height="{height}" rx="18" fill="#fff"/>']
        self.text(36, 47, title, 28, bold=True)
        self.text(36, 79, subtitle, 16, fill='#586c84')

    def text(self, x, y, lines, size=17, fill=None, bold=False, anchor='start', shape=False):
        if isinstance(lines, str):
            lines = [lines]
        attrs = f'x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'
        if fill:
            attrs += f' style="fill:{fill}"'
        if bold:
            attrs += ' font-weight="650"'
        if shape:
            attrs += ' class="shape"'
        spans = ''.join(f'<tspan x="{x}" dy="{0 if i == 0 else size+7}">{escape(line)}</tspan>'
                        for i, line in enumerate(lines))
        self.parts.append(f'<text {attrs}>{spans}</text>')

    def box(self, x, y, w, h, title, lines=(), kind='compute'):
        bg, border = COLORS[kind]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                          f'fill="{bg}" stroke="{border}" stroke-width="1.5"/>')
        self.text(x+16, y+29, title, 18, bold=True)
        if lines:
            self.text(x+16, y+56, lines, 16)

    def arrow(self, points, kind='compute', dashed=False):
        stroke = COLORS[kind][1]
        pts = ' '.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
                          f'stroke-width="2.2" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>')

    def note(self, lines):
        self.text(36, self.height-41, lines, 15, fill='#586c84')

    def render(self):
        return '\n'.join(self.parts + ['</svg>']) + '\n'


def diagrams(lang, configs):
    zh = lang == 'zh'
    def tr(cn, en):
        return cn if zh else en
    result = {}
    f = Figure(tr('模型结构地图：先看切分轴，再看缓存', 'Model map: hidden states, attention, and cache'),
               tr('同一张图比较六个主体型号；数字来自本栏固定配置。',
                  'Six pinned backbone configurations; extra prediction/vision modules are counted separately.'), 692)
    cards = [
        ('DeepSeek-R1', '61 × H7168', 'MLA', tr('3 Dense + 58 MoE', '3 dense + 58 MoE'), tr('每层 512 维 KV + 64 维 RoPE', 'Per layer: 512 KV + 64 RoPE dimensions')),
        ('GLM-5.1', '78 × H6144', 'MLA + DSA', tr('3 Dense + 75 MoE', '3 dense + 75 MoE'), tr('每层自有主 KV', 'Main KV owned by every layer')),
        ('GLM-5.2', '78 × H6144', 'MLA + IndexShare', tr('21 Full + 57 Shared', '21 Full + 57 Shared'), tr('共享索引；主 KV 仍自有', 'Shared indices; separate main KV')),
        ('V4-Flash', '43 × H4096', 'SWA + CSA / HCA', '256 experts · top-6', tr('四流 mHC；逐层缓存', 'Four-stream mHC; per-layer KV')),
        ('V4-Pro', '61 × H7168', 'SWA + CSA / HCA', '384 experts · top-6', tr('四流 mHC；逐层缓存', 'Four-stream mHC; per-layer KV')),
        ('V4.1-Flash', '40 × H5120', 'CED + CSA2', '384 experts · top-6', tr('4 个全局 KV 来源层', '4 global KV source layers')),
    ]
    for i, card in enumerate(cards):
        y = 112+i*82
        f.box(36, y, 175, 66, card[0], [card[1]], 'neutral')
        f.box(238, y, 278, 66, card[2], [card[3]], 'compute')
        f.box(544, y, 540, 66, card[4],
              [tr('残差 / Attention / FFN / KV 必须分别读 shape', 'Read residual, attention, FFN and cache shapes separately')], 'cache')
        f.arrow([(211,y+33),(238,y+33)])
        f.arrow([(516,y+33),(544,y+33)], 'cache')
    f.note(tr('额外模块：R1 / GLM / V4 各 1 层 MTP；V4.1 有 3 层 DSpark 与 32 层 ViT。',
              'Auxiliary modules: R1/GLM/V4 each have 1 MTP layer; V4.1 has 3 DSpark stages and a 32-layer ViT.'))
    result['model-map'] = f.render()

    f = Figure(tr('Shape 怎么读：投影、拼头与量化存储', 'Reading shapes: projection, head layout, and packed storage'),
               tr('以 GLM 为例；所有权重先按未切分的 [out,in] 理解。',
                  'GLM example. Weights below are unpartitioned logical [out,in] matrices.'), 510)
    f.box(36,130,245,120,tr('当前隐藏状态', 'Current hidden state'), ['[B,S,6144]', 'B=2, S=128: [2,128,6144]'])
    f.box(335,130,246,120,'q_a → RMSNorm', ['W: [2048,6144]', 'out: [B,S,2048]'])
    f.box(635,130,246,120,'q_b', ['W: [16384,2048]', 'out: [B,S,16384]'])
    f.arrow([(281,190),(335,190)]);f.arrow([(581,190),(635,190)])
    f.box(635,305,449,96,tr('只重排轴，不改变元素个数', 'Reshape the axes, preserve element count'), ['[B,S,64,256] ↔ [B,64,S,256]'], 'cache')
    f.arrow([(881,190),(1000,190),(1000,305)])
    f.box(36,305,545,96,tr('逻辑权重 ≠ 物理打包 shape', 'Logical weights and packed storage differ'),
          ['[2304,5120] → FP4 payload [2304,2560]', tr('两元素 / 字节；scale、padding 另算', 'Two values per byte; scales and padding are extra')], 'expert')
    f.note(tr('Decode 常把 S 改为 1；历史长度 T 不会随之变成 1。', 'Ordinary decode often sets S=1; retained history length T does not become 1.'))
    result['shape-flow'] = f.render()

    f = Figure(tr('GLM 一个主干层：Attention 与 FFN 的两次残差', 'One GLM backbone layer: attention and FFN residuals'),
               tr('层间输入 / 输出都是 [B,S,6144]；0–2 用 Dense，3–77 用 MoE。',
                  'Layer input/output: [B,S,6144]. Dense FFN at 0–2, MoE at 3–77.'), 590)
    stages=[(36,'h_l', ['[B,S,6144]']), (220,'RMSNorm', ['H=6144']),
            (404,'MLA + DSA',['Q: 64 × 256']), (588,tr('残差相加', 'Residual add'), ['+ h_l'])]
    for x,t,lines in stages:f.box(x,140,160,104,t,lines)
    for x in [196,380,564]:f.arrow([(x,192),(x+24,192)])
    f.arrow([(116,140),(116,111),(668,111),(668,140)],'neutral')
    f.box(588,330,160,104,'RMSNorm',['H=6144'])
    f.box(370,330,194,104,'Dense / MoE',['F=12288 / 2048'],'expert')
    f.box(152,330,194,104,tr('残差相加', 'Residual add'),['[B,S,6144]'])
    f.arrow([(668,244),(668,330)]);f.arrow([(588,382),(564,382)]);f.arrow([(370,382),(346,382)])
    f.arrow([(748,192),(800,192),(800,476),(249,476),(249,434)],'neutral')
    f.box(856,330,228,104,'h_(l+1)', ['[B,S,6144]'], 'cache')
    f.arrow([(249,434),(249,500),(970,500),(970,434)])
    f.note(tr('实线表示计算数据流；绕行的灰线表示残差。所有 shape 均未做 TP / EP 切分。',
              'Colored arrows carry computed states; gray bypasses carry residuals. Shapes precede TP/EP partitioning.'))
    result['glm-block'] = f.render()

    f = Figure(tr('GLM MLA：从 6144 维隐藏状态到 Q、K、V', 'GLM MLA: hidden state to Q, K, and V'),
               tr('图中是逻辑投影；展开的 K/V 不一定作为完整历史长期存储。',
                  'Logical projections. Expanded K/V need not be stored persistently for all history.'), 700)
    f.box(36,255,180,106,'X', ['[B,S,6144]'])
    f.box(274,128,242,104,'q_a → norm → q_b', ['6144 → 2048 → 16384', 'Q: [B,S,64,256]'])
    f.box(274,330,242,104,'kv_a', ['6144 → 576', '512 latent + 64 RoPE'], 'cache')
    f.arrow([(216,290),(242,290),(242,180),(274,180)])
    f.arrow([(216,330),(242,330),(242,382),(274,382)],'cache')
    f.box(575,285,242,114,'kv_b + RoPE', ['512 → 28672', 'K: [B,S,64,256]', 'V: [B,S,64,256]'], 'cache')
    f.arrow([(516,382),(545,382),(545,342),(575,342)],'cache')
    f.box(873,128,211,104,'Attention', ['Q × K → weights', 'weights × V'])
    f.arrow([(516,180),(873,180)])
    f.arrow([(817,342),(846,342),(846,205),(873,205)],'cache')
    f.box(575,490,242,104,tr('主潜在缓存', 'Main latent cache'), ['[B,T,512]', '+ [B,T,64]'], 'cache')
    f.arrow([(395,434),(395,542),(575,542)],'cache',True)
    f.box(873,490,211,104,'o_proj', ['16384 → 6144', '[B,S,6144]'])
    f.arrow([(978,232),(978,490)])
    f.note(tr('主 Q/K 头宽 = 192 NoPE + 64 RoPE；Indexer 的 32×128 是另一条旁路。',
              'Main Q/K width = 192 NoPE + 64 RoPE. The 32×128 indexer is a separate side path.'))
    result['glm-mla'] = f.render()

    f = Figure(tr('GLM MoE：选 8 个路由专家，同时计算共享专家', 'GLM MoE: top-8 routed experts plus one shared expert'),
               tr('所有专家输入宽度都是 6144；N_e 取决于本批实际路由。',
                  'Expert input width is 6144. N_e depends on actual routing assignments.'), 590)
    f.box(36,220,205,115,'Tokens', ['[N,6144]', 'N = B × S'])
    f.box(295,125,225,110,'Router', ['scores: [N,256]', 'ids / weights: [N,8]'], 'index')
    f.box(575,125,250,110,tr('256 中选 8', '8 selected from 256'), ['6144 → 2048 → 6144', '[N_e,6144]'], 'expert')
    f.box(575,330,250,110,tr('1 个共享专家', '1 shared expert'), ['6144 → 2048 → 6144', '[N,6144]'], 'expert')
    f.box(879,220,205,115,tr('加权合并 + 相加', 'Combine + add'), ['[N,6144]', '→ [B,S,6144]'])
    f.arrow([(241,250),(267,250),(267,180),(295,180)],'index')
    f.arrow([(520,180),(575,180)],'expert')
    f.arrow([(241,300),(267,300),(267,385),(575,385)],'expert')
    f.arrow([(825,180),(850,180),(850,250),(879,250)],'expert')
    f.arrow([(825,385),(850,385),(850,305),(879,305)],'expert')
    f.note(tr('仅层 3–77 用此 MoE；层 0–2 使用 F=12288 的 Dense FFN。',
              'This MoE applies to layers 3–77. Layers 0–2 use dense FFNs with F=12288.'))
    result['glm-moe'] = f.render()

    f = Figure(tr('IndexShare：共享选中位置，不共享各层主 KV', 'IndexShare: reuse selection, retain per-layer main KV'),
               tr('展示层 2–6；橙色虚线传 Top-k，横向实线传隐藏状态。',
                  'Layers 2–6. Dashed orange arrows carry top-k; solid arrows carry hidden states.'), 630)
    for row, name in enumerate(['GLM-5.1','GLM-5.2']):
        y=145+row*235;f.text(36,y-20,name,21,bold=True)
        for j, layer in enumerate(range(2,7)):
            x=36+j*215;own=row==0 or layer in (2,6)
            f.box(x,y,188,91,f'Layer {layer}',[tr('自己计算 Top-k','Compute top-k') if own else tr('复用层 2 Top-k','Reuse L2 top-k')], 'index' if own else 'neutral')
            f.box(x,y+112,188,57,f'KV {layer}',[], 'cache')
            if j<4:f.arrow([(x+188,y+46),(x+215,y+46)])
            if row==1 and j in (1,2,3):f.arrow([(130,y),(130,y-51),(x+94,y-51),(x+94,y)],'index',True)
    f.note(tr('GLM-5.2 全模型为 21 个 Full Indexer + 57 个 Shared 层；78 层主潜在 KV 仍各自持有。',
              'GLM-5.2: 21 Full indexers + 57 Shared layers; all 78 layers retain their own main latent KV.'))
    result['index-share'] = f.render()

    f = Figure(tr('361 个主干层的结构色带', 'A structural map of 361 backbone layers'),
               tr('每个小格是一层；打开原图后悬停查看层号。分类标在各色带下方。',
                  'Open the image to inspect layer-index tooltips. Categories are labeled below each strip.'), 725)
    rows=[]
    for slug in ('deepseek-r1','glm-5-1','glm-5-2','deepseek-v4-flash','deepseek-v4-pro','deepseek-v4-1-flash'):
        raw=configs[slug];c=raw.get('text_config',raw);n=c['num_hidden_layers']
        if slug in ('deepseek-r1','glm-5-1'):kinds=['compute']*c['first_k_dense_replace']+['expert']*(n-c['first_k_dense_replace']);label='Dense / MoE'
        elif slug=='glm-5-2':kinds=['index' if v=='full' else 'neutral' for v in c['indexer_types']];label='Full / Shared'
        elif slug.startswith('deepseek-v4-1'):
            kinds=['neutral' if c['compress_ratios'][i]==0 else 'cache' if i in c['kv_source_layer_ids'] else 'index' if i in c['index_source_layer_ids'] else 'compute' for i in range(n)];label='SWA / Full / Reindex / Reuse'
        else:kinds=['neutral' if r==0 else 'compute' if r==4 else 'index' for r in c['compress_ratios'][:n]];label='SWA / CSA / HCA'
        rows.append((slug,n,kinds,label))
    for row,(slug,n,kinds,label) in enumerate(rows):
        y=145+row*85;display=slug.replace('deepseek-','').upper().replace('GLM-5-','GLM-5.')
        f.text(36,y,display,18,bold=True);f.text(36,y+25,f'{n} layers',15)
        cell=800/n
        for i,kind in enumerate(kinds):
            bg,stroke=COLORS[kind];x=250+i*cell
            if slug in ('deepseek-r1','glm-5-1'):
                mode='Dense FFN' if i<3 else 'MoE FFN'
            elif slug=='glm-5-2':
                mode='Full indexer' if kind=='index' else 'Shared top-k'
            elif slug=='deepseek-v4-1-flash':
                mode={'neutral':'SWA only','cache':'Full KV source','index':'Reindex','compute':'Reuse'}[kind]
            else:
                mode={'neutral':'SWA only','compute':'SWA + CSA (r=4)','index':'SWA + HCA (r=128)'}[kind]
            f.parts.append(f'<rect x="{x:.3f}" y="{y-18}" width="{cell-1:.3f}" height="31" rx="2" fill="{bg}" stroke="{stroke}"><title>Layer {i}: {escape(mode)}</title></rect>')
            if i==0 or i==n-1 or i%10==0:f.text(x+cell/2,y+33,str(i),12,anchor='middle')
        f.text(250,y+57,label,14,fill='#586c84')
    f.note(tr('蓝：Dense / CSA / Reuse；紫：MoE；橙：Full Indexer / HCA / Reindex；绿：KV Full；灰：Shared / SWA。',
              'Blue: Dense/CSA/Reuse; purple: MoE; amber: Full indexer/HCA/Reindex; green: KV Full; gray: Shared/SWA.'))
    result['layer-map'] = f.render()

    f = Figure(tr('mHC：四流残差与单流计算分支', 'mHC: four residual streams, one sublayer input'),
               tr('适用于 V4 / V4.1 的 shape 示意；混合系数的计算时序请看各模型正文。',
                  'V4 / V4.1 shape view; coefficient scheduling differs between model generations.'), 580)
    f.box(36,135,245,280,tr('层间残差', 'Residual stream'), ['[B,S,4,H]', '', 'stream 0: [B,S,H]', 'stream 1: [B,S,H]', 'stream 2: [B,S,H]', 'stream 3: [B,S,H]'], 'neutral')
    f.box(335,135,230,110,tr('pre 加权合并', 'pre: weighted collapse'), ['4 streams → 1', '[B,S,H]'])
    f.box(620,135,230,110,'Attention / FFN', ['input: [B,S,H]', 'output: [B,S,H]'])
    f.box(620,330,230,110,tr('post + 4×4 混合', 'post + 4×4 mixing'), ['[B,S,H] → [B,S,4,H]'], 'cache')
    f.box(900,330,184,110,tr('下一子层', 'Next sublayer'), ['[B,S,4,H]'], 'cache')
    f.arrow([(281,190),(335,190)]);f.arrow([(565,190),(620,190)])
    f.arrow([(735,245),(735,330)]);f.arrow([(281,385),(620,385)],'neutral')
    f.arrow([(850,385),(900,385)],'cache')
    f.note(tr('H：Flash=4096，Pro=7168，V4.1=5120；四流并不把每个专家的输入宽度变成 4H。',
              'H: Flash=4096, Pro=7168, V4.1=5120. Expert input width remains H, not 4H.'))
    result['mhc'] = f.render()

    f = Figure(tr('DeepSeek V4 注意力：窗口 + 全局路径 + 分组 O', 'DeepSeek V4 attention: window, global path, grouped O'),
               tr('图中 Q 按全局头数 A 表示；缓存是所有头共享的 512 维潜在向量。',
                  'A is the global query-head count; KV uses shared 512-dimensional latent vectors.'), 650)
    f.box(36,245,184,100,'X', ['[B,S,H]'])
    f.box(274,125,245,110,'wq_a → norm → wq_b', ['H → Rq → A×512', 'Q: [B,S,A,512]'])
    f.box(274,340,245,110,'wkv / compressor', ['window: [B,128,512]', 'global: [B,T/r,512]'], 'cache')
    f.box(575,340,245,110,'SWA + CSA / HCA', ['r=4: sparse top-k', 'r=128: compressed'], 'index')
    f.box(875,125,209,110,'Attention', ['[B,S,A,512]'])
    f.box(875,340,209,140,tr('分组 O 投影', 'Grouped O projection'), ['[B,S,G,4096]', '→ [B,S,G,1024]', '→ [B,S,H]'])
    f.arrow([(220,270),(245,270),(245,180),(274,180)])
    f.arrow([(220,320),(245,320),(245,395),(274,395)],'cache')
    f.arrow([(519,395),(575,395)],'index');f.arrow([(519,180),(875,180)])
    f.arrow([(820,395),(846,395),(846,206),(875,206)],'index')
    f.arrow([(979,235),(979,340)])
    f.note([tr('Flash: H4096 / A64 / Rq1024 / G8；Pro: H7168 / A128 / Rq1536 / G16。',
               'Flash: H4096 / A64 / Rq1024 / G8. Pro: H7168 / A128 / Rq1536 / G16.'),
            tr('r=0 的层只有 SWA；T/r 表示完成压缩组数，精确长度取 floor(T/r)。',
               'r=0 layers use SWA only. Compressed valid length is floor(T/r).')])
    result['v4-attention'] = f.render()

    f = Figure(tr('V4.1 全局 KV：4 个所有者，多层使用', 'V4.1 global KV: four owners, many consumers'),
               tr('每个层仍有自己的 SWA 窗口；此图只画全局 KV 和 Indexer 的共享。',
                  'Each layer also owns an SWA window. This figure isolates global KV and selection sharing.'), 690)
    groups=[(2,'2–7','floor(T/2)', '2'),(8,'8–13','floor(T/2)','8'),(14,'14–19','floor(T/2)','14'),(20,'20–39','T','20,24,28,32,36')]
    for row,(owner,consumers,length,indexers) in enumerate(groups):
        y=115+row*118
        f.box(36,y,207,91,f'Full: L{owner}', [f'r={2 if owner<20 else 1}'], 'cache')
        f.box(290,y,365,91,tr('同一组主 KV + Indexer Key', 'One main-KV + index-key group'), [f'[B,{length},512] + 128-d K'], 'cache')
        f.box(705,y,379,91,tr(f'使用层 {consumers}', f'Consumers: layers {consumers}'), [f'Indexer: L{indexers}'], 'index')
        f.arrow([(243,y+45),(290,y+45)],'cache');f.arrow([(655,y+45),(705,y+45)],'cache')
    f.box(36,603,1048,45,tr('层 0–1：仅 SWA；层 20 来源于 Encoder 输出；24/28/32/36 重新选择位置，但不新建全局 KV。',
                            'L0–1: SWA only. L20 reads encoder output; L24/28/32/36 reindex without a new global KV group.'),[], 'neutral')
    result['v41-kv'] = f.render()

    f = Figure(tr('V4.1 视觉：Patch → ViT → 语言空间', 'V4.1 vision: patches to the language embedding space'),
               tr('单图逻辑 shape；P=n_h×n_w，P′=ceil(n_h/3)×ceil(n_w/3)。',
                  'Single-image logical shapes. P=n_h×n_w; P′=ceil(n_h/3)×ceil(n_w/3).'), 550)
    stages=[(36,'14×14 RGB patch',['[P,588]', '588 = 3×14×14']),
            (310,tr('Patch 投影','Patch projection'),['W: [1024,588]', '[P,1024]']),
            (584,'32 × ViT', ['Q/K/V: [P,16,64]', 'FFN F=2816']),
            (858,tr('3×3 下采样','3×3 downsample'),['[P′,9216]', '9216 = 9×1024'])]
    for x,t,lines in stages:f.box(x,145,226,120,t,lines)
    for x in (262,536,810):f.arrow([(x,205),(x+48,205)])
    f.box(584,345,500,108,tr('两层 Aligner → 替换语言序列中的图片位置', 'Two-layer aligner → image positions in language input'),
          ['9216 → 5120 → 5120', '[P′,5120]'], 'cache')
    f.arrow([(971,265),(971,345)],'cache')
    f.box(36,345,500,108,tr('每个 ViT 层保留的宽度', 'Width preserved by every ViT layer'),
          ['[P,1024] → Attention + FFN → [P,1024]', tr('图像内双向 Attention + 2D RoPE', 'Bidirectional image attention + 2D RoPE')], 'neutral')
    f.note(tr('视觉 Token 数随分辨率、缩放和边界 padding 改变；不是每张图片固定一个 P。',
              'Visual token counts depend on resolution, resizing and padding; P is not constant across images.'))
    result['v41-vision'] = f.render()

    f = Figure(tr('V4.1 Engram：24 次稀疏查表，写回四流残差', 'V4.1 Engram: 24 sparse lookups feed the residual stream'),
               tr('仅在层 1 和 14；图中是一个位置的查表维度，不遍历全部表项。',
                  'At layers 1 and 14 only. Lookup dimensions do not imply scanning every table entry.'), 620)
    f.box(36,140,230,110,'2 / 3 / 4-gram', ['3 lengths × 8 heads', 'hash IDs: [B,S,24]'], 'memory')
    f.box(320,140,230,110,tr('稀疏 Embedding 查表','Embedding lookup'), ['[B,S,24,256]', '→ [B,S,6144]'], 'memory')
    f.box(604,140,230,110,'wkv', ['W: [25600,6144]', 'out: [B,S,25600]'], 'memory')
    f.arrow([(266,195),(320,195)],'memory');f.arrow([(550,195),(604,195)],'memory')
    f.box(320,355,230,110,'Key', ['[B,S,4,5120]'], 'index')
    f.box(604,355,230,110,'Value', ['[B,S,5120]'], 'cache')
    f.arrow([(719,250),(719,300),(435,300),(435,355)],'memory')
    f.arrow([(719,250),(719,355)],'memory')
    f.box(888,355,196,110,tr('门控写回','Gated merge'), ['[B,S,4,5120]'])
    f.arrow([(834,410),(888,410)],'cache');f.arrow([(550,410),(576,410),(576,500),(986,500),(986,465)],'index')
    f.box(36,355,230,110,tr('当前残差流','Current residual'), ['[B,S,4,5120]'], 'neutral')
    f.arrow([(151,465),(151,535),(1030,535),(1030,465)],'neutral')
    f.note(tr('Key 与当前残差决定 gate；同一 Value 按四流 gate 写入。查表参数不等于每 Token 激活参数。',
              'Key/residual matching determines gates; a shared value updates four streams. Stored and active parameters differ.'))
    result['v41-engram'] = f.render()

    f = Figure(tr('V4.1 DSpark：草稿网络与主干验证分开', 'V4.1 DSpark: drafting and backbone verification'),
               tr('40 层主干之外有 3 层草稿网络；块大小 5 不等于一定接受 5 个 Token。',
                  'Three draft stages are outside the 40-layer backbone. Block size 5 does not imply 5 accepted tokens.'), 590)
    f.box(36,135,260,110,tr('主干目标层 37 / 38 / 39','Target layers 37 / 38 / 39'), ['concat: [B,S,15360]', 'main_proj: 15360 → 5120'])
    f.box(36,335,260,110,tr('草稿块 Embedding','Draft-block embedding'), ['[B,5,5120]', '→ [B,5,4,5120]'])
    f.box(370,335,270,110,'DSpark 40 / 41 / 42', ['128 experts · top-3', tr('读取 main_x 上下文','Conditioned on main_x')], 'expert')
    f.box(714,335,370,110,tr('草稿词表头 → 主干验证','Draft head → backbone verification'), ['logits: [B,5,129280]', tr('接受前缀由验证结果决定','Accepted prefix follows verification')], 'cache')
    f.arrow([(296,390),(370,390)],'expert')
    f.arrow([(296,190),(505,190),(505,335)])
    f.arrow([(640,390),(714,390)],'cache')
    f.note(tr('主干层数 40、草稿层数 3、草稿块大小 5、接受数：四个不同计数。',
              'Backbone depth 40, draft depth 3, block size 5 and accepted count are distinct quantities.'))
    result['v41-dspark'] = f.render()

    f = Figure(tr('V4.1 全局 KV：890 字节 / 原始 Token 的来源', 'V4.1 global KV: where 890 bytes per original token come from'),
               tr('仅打包后的全局载荷；不包括 SWA、页元数据、缓冲区和工作区。',
                  'Packed global payload only; SWA, page metadata, buffers and workspaces are additional.'), 540)
    f.box(36,132,1048,78,tr('缓存密度：层 2 / 8 / 14 各压缩 2 倍，层 20 比例为 1',
                           'Cache density: layers 2 / 8 / 14 use ratio 2; layer 20 uses ratio 1'),
          ['3 × (1/2) + 1 = 2.5 vectors / original token'], 'neutral')
    f.box(36,260,630,120,tr('主 KV = 720 bytes','Main KV = 720 bytes'), ['(512/2 + 512/16) × 2.5', 'FP4 payload + 1-byte scale / 16 values'], 'cache')
    f.box(690,260,394,120,tr('Indexer K = 170 bytes','Index keys = 170 bytes'), ['(128/2 + 128/32) × 2.5', '1-byte scale / 32 values'], 'index')
    f.text(560,447,'720 + 170 = 890 bytes / token',28,bold=True,anchor='middle')
    f.note(tr('890 已经合计四组全局缓存，不再乘 40 层；常驻 BF16 主干 SWA 窗口另约 5 MiB / 序列。',
              '890 already sums the four global groups: do not multiply by 40. BF16 backbone SWA adds about 5 MiB/sequence.'))
    result['v41-cache'] = f.render()

    f = Figure(tr('SWA、CSA、HCA：比较访问路径', 'SWA, CSA, HCA: compare access paths'),
               tr('V4 的 CSA / HCA 层也有 SWA 分支；下面分别追踪局部与全局路径。',
                  'V4 CSA/HCA layers also retain SWA. Trace local and global paths separately.'), 650)
    columns = [
        ('SWA', 'cache',
         (tr('最近原始位置', 'Recent raw positions'), ['[B,128,512]', tr('最多 min(128,T) 个位置', 'At most min(128,T) positions')]),
         (tr('按窗口规则读取', 'Read by window rule'), [tr('只保留因果可见的最近位置', 'Recent causally visible positions'), tr('无需学习式 Top-k', 'No learned top-k selection')]),
         (tr('主注意力的局部分支', 'Local main-attention branch'), [tr('保持近期信息的细节', 'Preserves recent detail')])),
        ('CSA', 'compute',
         (tr('全局历史压缩 4 倍', 'Compress global history by 4'), ['[B,floor(T/4),512]', 'Indexer K: 128 dimensions']),
         (tr('Indexer 评分 → Top-k', 'Indexer scores → top-k'), ['Flash: k ≤ 512', 'Pro: k ≤ 1024']),
         (tr('选中的全局位置 + SWA', 'Selected global entries + SWA'), [tr('较细压缩；稀疏访问', 'Finer compression; sparse reads')])),
        ('HCA', 'index',
         (tr('全局历史压缩 128 倍', 'Compress global history by 128'), ['[B,floor(T/128),512]']),
         (tr('读取全部有效压缩条目', 'Read all valid compressed entries'), [tr('不实例化 CSA Indexer', 'No CSA indexer instance')]),
         (tr('全部全局压缩条目 + SWA', 'All compressed entries + SWA'), [tr('更粗压缩；保留全局覆盖', 'Coarser global representation')])),
    ]
    for i,(name,kind,*stages) in enumerate(columns):
        x=36+i*360;f.text(x,125,name,23,bold=True)
        for y,(title,lines) in zip((155,300,445),stages,strict=True):
            f.box(x,y,328,105,title,lines,kind)
        f.arrow([(x+164,260),(x+164,300)],kind)
        f.arrow([(x+164,405),(x+164,445)],kind)
    f.note(tr('Indexer 决定“去哪里读”；主注意力负责读取主 KV 并完成最终内容加权。',
              'The indexer chooses where to read. Main attention reads main KV and computes the final weighted result.'))
    result['attention-paths'] = f.render()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    configs = load_configs()
    # Numeric annotations are deliberately authored for these pinned structures.
    # Reject incompatible updates rather than silently publishing stale shapes.
    for slug in ('deepseek-r1','glm-5-1','glm-5-2'):
        expected = dict(num_hidden_layers=61, hidden_size=7168, num_attention_heads=128,
                        q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
                        qk_rope_head_dim=64, v_head_dim=128, n_routed_experts=256,
                        num_experts_per_tok=8, moe_intermediate_size=2048, intermediate_size=18432,
                        first_k_dense_replace=3) if slug == 'deepseek-r1' else dict(
                        num_hidden_layers=78, hidden_size=6144, num_attention_heads=64,
                        q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
                        qk_rope_head_dim=64, v_head_dim=256, n_routed_experts=256,
                        num_experts_per_tok=8, moe_intermediate_size=2048, intermediate_size=12288)
        for key,value in expected.items():
            if configs[slug][key]!=value:
                raise ValueError(f'{slug}/{key}: revise diagram annotations for the new config')
    for slug,depth,width,heads,qrank,groups,experts,ffn in (
        ('deepseek-v4-flash',43,4096,64,1024,8,256,2048),
        ('deepseek-v4-pro',61,7168,128,1536,16,384,3072),
        ('deepseek-v4-1-flash',40,5120,64,1280,8,384,2304),
    ):
        c=configs[slug].get('text_config',configs[slug])
        expected=dict(num_hidden_layers=depth,hidden_size=width,num_attention_heads=heads,
                      q_lora_rank=qrank,o_groups=groups,n_routed_experts=experts,
                      moe_intermediate_size=ffn,hc_mult=4,head_dim=512,sliding_window=128,
                      o_lora_rank=1024,num_experts_per_tok=6)
        for key,value in expected.items():
            if c[key]!=value:
                raise ValueError(f'{slug}/{key}: revise diagram annotations for the new config')
    stale=[]
    count=0
    for lang in ('en','zh'):
        for name,svg in diagrams(lang,configs).items():
            path=OUT/lang/(name+'.svg');count+=1
            if args.write:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(svg.encode('utf-8'))
            elif not path.is_file() or path.read_bytes()!=svg.encode('utf-8'):
                stale.append(str(path.relative_to(ROOT)))
    if stale:
        raise SystemExit('Missing/outdated diagrams; run --write:\n'+'\n'.join(stale))
    print(f'PASS: {count} reproducible bilingual SVG model diagrams.')


if __name__ == '__main__':
    main()
