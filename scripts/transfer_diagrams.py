"""Generate bilingual diagrams for a concrete RDMA transaction and SQE boundaries."""
import argparse
from pathlib import Path

from model_diagrams import Figure

OUT = Path(__file__).resolve().parents[1] / 'docs/assets/transfer-diagrams'


def diagrams(lang):
    def tr(zh, en):
        return zh if lang == 'zh' else en

    f = Figure(tr('一次 64 KiB Write with Immediate：从入队到消费',
                  'One 64 KiB Write with Immediate: post to consume'),
               tr('RC · 非 Inline · 已注册 Host 内存；两侧 CQ 独立推进，图中未规定 Poll 的先后。',
                  'RC, non-inline, registered host memory; the two applications poll independently.'), 850)
    actors = [(140, tr('应用 A / SQ / CQ', 'App A / SQ / CQ')),
              (410, tr('网卡 A', 'NIC A')), (710, tr('网卡 B', 'NIC B')),
              (980, tr('应用 B / RQ / CQ', 'App B / RQ / CQ'))]
    for x, label in actors:
        f.box(x-125, 103, 250, 48, label, kind='neutral')
        f.parts.append(f'<path d="M{x} 154 V665" stroke="#bdc9d9" stroke-dasharray="5 5"/>')

    def send(x1, x2, y, label, kind='compute', dashed=False):
        f.arrow([(x1, y), (x2, y)], kind, dashed)
        f.text((x1+x2)/2, y-12, label, 15, anchor='middle')

    send(980, 710, 197, tr('① 预贴接收 WR → RQ WQE', '1. Post receive WR → RQ WQE'))
    send(980, 140, 248, tr('② 控制通道：目标地址、rkey、长度；接收资源已就绪',
                         '2. Control channel: address, rkey, length; receiver ready'), dashed=True)
    send(140, 410, 312, tr('③ 源已就绪；WR → SQ WQE → Doorbell',
                         '3. Source ready; WR → SQ WQE → doorbell'))
    f.box(283, 334, 254, 59, tr('④ DMA 读源缓冲区', '4. DMA-read source'), kind='cache')
    send(410, 710, 434, tr('⑤ 网络分包：64 KiB 载荷', '5. Packetized 64 KiB payload'), 'cache')
    send(710, 980, 500, tr('⑥ DMA 写目标；接收 CQE', '6. DMA to target; receive CQE'), 'cache')
    send(710, 410, 559, tr('⑦ RC 传输确认 ACK', '7. RC transport ACK'), dashed=True)
    send(410, 140, 620, tr('⑧ 本地发送 CQE', '8. Local send CQE'))
    f.box(24, 684, 510, 94, tr('A：Poll → 检查成功 WC → 复用源',
                             'A: poll → check success WC → reuse source'),
          [tr('wr_id=42；不证明 B 的计算已结束。',
              'wr_id=42; does not establish that B finished computing.')], 'compute')
    f.box(560, 684, 536, 94, tr('B：Poll → 检查批次 → 消费 → 归还 credit',
                             'B: poll → check generation → consume → credit'),
          [tr('wr_id=900；imm=7；credit 是额外应用协议。',
              'wr_id=900; imm=7; credit uses an extra application protocol.')], 'cache')
    f.note(tr('图示成功路径；WR / WQE / 网络包 / CQE 并非一一对应；GPU 目标还需设备可见性保证。',
              'Successful path; WRs, WQEs, packets, and CQEs are not one-to-one. GPU targets need device visibility.'))

    g = Figure(tr('SQE 在哪里：设备任务调度与网卡工作队列',
                  'Where SQEs fit: device scheduling and NIC work queues'),
               tr('上图是 Host verbs；下图只解释采用设备任务调度 + RDMA 后端时的职责边界。',
                  'The transaction uses host verbs. This map shows a device-scheduled RDMA backend.'), 670)
    g.box(36, 119, 320, 106, tr('框架 / 通信算子', 'Framework / communication op'),
          [tr('请求、Stream 依赖、分块计划', 'Request, stream dependencies, chunks'),
           tr('例：发送一层 KV', 'Example: transfer one layer of KV')])
    g.box(400, 119, 320, 106, tr('Runtime / 设备任务队列', 'Runtime / device task queue'),
          [tr('任务描述符：SQE', 'Task descriptor: SQE'),
           tr('记录调度与执行依赖', 'Scheduling and execution dependencies')])
    g.box(764, 119, 320, 106, tr('设备调度 / 执行后端', 'Device scheduler / execution'),
          [tr('等待前置任务；执行通信工作', 'Wait for producer; execute comm work'),
           tr('实现随产品与模式变化', 'Product- and mode-specific')])
    g.arrow([(356, 173), (400, 173)])
    g.arrow([(720, 173), (764, 173)])
    g.arrow([(924, 225), (924, 294)])
    g.box(764, 294, 320, 106, tr('RDMA 通信后端', 'RDMA communication backend'),
          [tr('编码网卡 SQ 中的 WQE', 'Encode WQEs in NIC SQ'),
           tr('发布记录并通知 NIC', 'Publish records and notify NIC')], 'index')
    g.box(400, 294, 320, 106, tr('NIC / DMA 引擎', 'NIC / DMA engine'),
          [tr('取 WQE → 读源 → 网络 → 写目标', 'WQE → source → network → target'),
           tr('完成信息写入对应 CQ', 'Write completion into associated CQ')], 'cache')
    g.box(36, 294, 320, 106, tr('通信完成与任务推进', 'Communication / task progress'),
          [tr('检查 CQE / WC 与通知', 'Check CQE / WC and notifications'),
           tr('满足依赖后推进后续任务', 'Release subsequent dependent work')], 'cache')
    g.arrow([(764, 347), (720, 347)], 'cache')
    g.arrow([(400, 347), (356, 347)], 'cache')
    g.box(36, 443, 1048, 133, tr('读图边界', 'Scope of this map'),
          [tr('一个算子可拆成多个任务、多个 WQE；不能按 1 个 SQE = 1 个 WQE = 1 个 CQE 计数。',
              'An operator can expand into multiple tasks and WQEs; do not assume a 1:1:1 mapping.'),
           tr('GPU 直接提交可从内核进入 NIC 队列；Host verbs 可直接经 provider 提交。',
              'GPU-direct submission can enter NIC queues from a kernel; host verbs use a provider.'),
           tr('设备调度的 SQ 与 RDMA 的 Send Queue 必须分别标明所属模块。',
              'Label the subsystem: a device scheduling SQ and an RDMA send queue are distinct.')], 'neutral')
    g.note(tr('职责示意，不代表所有 CANN / HCCL 版本的固定内部调用链。',
              'A responsibility map, not a fixed internal call chain for every CANN / HCCL release.'))
    return {'write-with-immediate': f.render(), 'sqe-to-wqe': g.render(), **jetty_diagrams(lang)}


def jetty_diagrams(lang):
    def tr(zh, en):
        return zh if lang == 'zh' else en

    f = Figure(tr('URMA 双边传输：A 的 JFS 到 B 的 JFR',
                  'URMA Send/Recv: JFS on A to JFR on B'),
               tr('以 4 KiB SEND 为例；资源预先创建，ID 为示意值；普通单边 WRITE 不沿用接收流程。',
                  'One 4 KiB SEND; resources already exist and IDs are illustrative. Ordinary WRITE differs.'), 820)
    f.box(36, 117, 440, 141, tr('A：Jetty / 发送侧 JFS', 'A: Jetty / send side JFS'),
          ['local ID = 17; user_ctx = 42',
           tr('发送 WR：SEND、src、长度、对端句柄', 'Send WR: SEND, src, length, target handle'),
           tr('provider 编码 SQ 中的 WQE', 'Provider encodes WQE in SQ')])
    f.box(644, 117, 440, 141, tr('B：Jetty / 接收侧 JFR', 'B: Jetty / receive side JFR'),
          ['local ID = 29; user_ctx = 900',
           tr('先贴接收 WR：dst、容量、注册段', 'Prepost receive: dst, capacity, segment'),
           tr('provider 发布 RQ 中的接收描述', 'Provider publishes receive descriptor')])
    f.arrow([(476, 185), (644, 185)])
    f.text(560, 166, tr('导入的对端', 'Imported peer'), 15, anchor='middle')
    f.box(36, 312, 440, 103, tr('源载荷：src / 4096 字节', 'Source payload: src / 4096 bytes'),
          [tr('生产完成后，硬件读取源内容', 'Hardware reads after producer readiness'),
           tr('WQE 不是这 4096 字节本身', 'The WQE is not the 4096-byte payload')], 'cache')
    f.box(644, 312, 440, 103, tr('目标载荷：B 已贴的 dst', 'Destination: B preposted dst'),
          [tr('硬件按接收描述放置数据', 'Hardware places data using receive WR'),
           tr('SEND 不通过远端 dst 指针选缓冲区', 'SEND does not select a remote dst pointer')], 'cache')
    f.arrow([(256, 258), (256, 312)])
    f.arrow([(864, 258), (864, 312)])
    f.arrow([(476, 363), (644, 363)], 'cache')
    f.text(560, 344, tr('UB 传输', 'UB transfer'), 15, anchor='middle')
    f.box(36, 471, 440, 125, tr('A：关联 JFC / CQ（例 ID=3）', 'A: associated JFC / CQ (e.g. ID=3)'),
          [tr('按完成策略，硬件写发送 CQE', 'Hardware writes send CQE per policy'),
           'poll_jfc → CR: status, user_ctx=42',
           tr('检查成功后，按语义复用源', 'Check success before source reuse')], 'index')
    f.box(644, 471, 440, 125, tr('B：关联 JFC / CQ（例 ID=8）', 'B: associated JFC / CQ (e.g. ID=8)'),
          [tr('硬件写接收 CQE；不是放入整个任务', 'Hardware writes a receive CQE, not a task'),
           'poll_jfc → CR: status, user_ctx=900',
           tr('校验长度 / 批次，再满足设备可见性', 'Validate length / generation and visibility')], 'index')
    f.arrow([(256, 415), (256, 471)], 'cache')
    f.arrow([(864, 415), (864, 471)], 'cache')
    f.box(36, 650, 1048, 94, tr('上层同步与资源交接', 'Upper-layer synchronization and handoff'),
          [tr('A 本地发送完成、B 接收完成、Stream 等待返回、B 消费结束，是不同的证据。',
              'A send completion, B receive completion, stream wait, and B consumption are distinct evidence.'),
           tr('B 消费后归还目标槽位；两侧的 user_ctx 来自各自 WR，不自动跨端复制。',
              'Return target-slot credit after consumption; each user_ctx comes from its own local WR.')], 'neutral')
    f.note(tr('Jetty 可组合发送与接收资源；也可用独立 JFS / JFR。共享 JFC 时必须按请求身份分派 CR。',
              'Use Jetty or standalone JFS/JFR resources. Dispatch records by request identity when a JFC is shared.'))

    g = Figure(tr('同一个 Copy Task：WQE 如何进入 Runtime SQ',
                  'One copy task: WQE placement in Runtime SQ'),
               tr('CANN Runtime 公开代码快照；左侧 Direct WQE 与右侧 Doorbell 是两条不同路径。',
                  'Pinned public CANN Runtime snapshot: Direct WQE and doorbell are separate paths.'), 850)
    g.box(36, 113, 1048, 93, tr('Runtime 创建拷贝任务：src、dst、size、Stream 依赖',
                              'Runtime creates a copy task: src, dst, size, stream dependencies'),
          [tr('驱动准备 WQE / Jetty 信息，Runtime 再构造调度 SQE；任务在执行前进入调度。',
              'Driver prepares WQE / Jetty information; Runtime constructs scheduling SQEs before execution.')])
    g.arrow([(285, 206), (285, 250)])
    g.arrow([(835, 206), (835, 250)])
    g.box(36, 250, 498, 126, tr('路径一：Direct WQE', 'Path 1: Direct WQE'),
          [tr('Driver 返回 WQE 字节与长度', 'Driver returns WQE bytes and length'),
           'dieId / functionId / jettyId',
           tr('Runtime 保存到 task.ubDma', 'Runtime saves these in task.ubDma')])
    g.box(586, 250, 498, 126, tr('路径二：Doorbell', 'Path 2: Doorbell'),
          [tr('转换 WQE → 填 Jetty 队列', 'Convert WQEs → fill Jetty queue'),
           tr('查询 / 绑定 Jetty，记录 PI', 'Query / bind Jetty; track PI'),
           tr('WQE 数据已放在通信队列', 'WQE data resides in communication queue')], 'index')
    g.arrow([(285, 376), (285, 421)])
    g.arrow([(835, 376), (835, 421)], 'index')
    g.box(36, 421, 498, 75, tr('Runtime SQ 槽位 n：头 SQE', 'Runtime SQ slot n: header SQE'),
          ['ASYNCDMA + Direct mode + Jetty IDs'])
    g.box(36, 502, 498, 75, tr('Runtime SQ 槽位 n+1：WQE 数据', 'Runtime SQ slot n+1: WQE bytes'),
          [tr('硬件按 Direct 格式消费后继槽位', 'Hardware consumes the trailing direct payload')], 'cache')
    g.box(586, 421, 498, 156, tr('Runtime SQ：Doorbell SQE', 'Runtime SQ: doorbell SQE'),
          ['UBDMA + Doorbell mode',
           'dieId / functionId / jettyId / PI',
           tr('触发已填好的 Jetty 工作队列', 'Trigger the already populated Jetty queue'),
           tr('此模式无需把 WQE 附在下一槽', 'No trailing WQE slot for this mode')], 'index')
    g.arrow([(285, 577), (285, 635)], 'cache')
    g.arrow([(835, 577), (835, 635)], 'cache')
    g.box(36, 635, 1048, 111, tr('设备调度与搬运', 'Device scheduling and movement'),
          [tr('前置依赖满足 → 消费 Runtime SQE → 执行 Direct / Doorbell 路径 → UB 引擎搬运。',
              'Dependencies ready → consume Runtime SQE → direct / doorbell execution → UB movement.'),
           tr('JFC 通信完成与 Runtime 任务进度的映射由后端确定；不能直接等同两层完成。',
              'The backend maps JFC completion to Runtime task progress; the milestones are not equivalent.'),
           tr('Runtime SQ 是队列，SQE 是条目；一个 Copy Task 的物理条目数取决于所选模式。',
              'Runtime SQ is a queue, SQE an entry; the physical entry count depends on the selected mode.')], 'neutral')
    g.note(tr('模式分支与字段依据固定版本源码，不代表所有产品、软件版本或跨服务器路径都适用。',
              'Modes and fields follow a pinned implementation, not every product, release, or inter-server path.'))

    h = Figure(tr('“CQ OK”到“同步成功”：到底等待哪件事',
                  'From CQ success to synchronization: what was awaited?'),
               tr('每一行回答不同问题；B 的一个成功 CR 不会自动完成 A 的整个 Stream。',
                  'Each row answers a different question. One successful CR on B does not finish A\'s stream.'), 650)
    rows = [
        (120, tr('提交成功', 'Posting accepted'),
         [tr('请求进入队列', 'Request queued'), tr('仍可能执行失败', 'Execution can still fail')],
         tr('完成条件尚未满足', 'Completion still pending'),
         [tr('继续跟踪本次请求', 'Track this request'), tr('不能据此复用缓冲区', 'Do not infer buffer reuse')], 'compute'),
        (260, tr('A 的通信完成', 'A communication completion'),
         [tr('匹配请求，检查成功 CR', 'Match request; check success CR'), tr('按操作语义判断源可复用', 'Apply operation reuse rules')],
         tr('A 的 Runtime / Stream 等待', 'A Runtime / stream wait'),
         [tr('后端接通任务完成与依赖', 'Backend tracks task completion'), tr('等待范围内的任务全部满足', 'All work in wait scope completes')], 'index'),
        (400, tr('B 的接收完成', 'B receive completion'),
         [tr('只用于有接收通知的操作', 'Only for notified operations'), tr('校验身份、长度与批次', 'Check identity, length, generation')],
         tr('B 可消费 → 消费完 → 归还', 'B ready → consumed → credit'),
         [tr('先满足 HBM 可见性与依赖', 'Establish HBM visibility first'), tr('消费后才归还目标槽位', 'Release target after consumption')], 'cache'),
    ]
    for y, left, left_lines, right, right_lines, kind in rows:
        h.box(36, y, 480, 109, left, left_lines, kind)
        h.box(604, y, 480, 109, right, right_lines, kind)
        h.arrow([(516, y+54), (604, y+54)], kind)
    h.note([tr('通信完成与 Runtime 完成需按后端映射；同步只确认 API 约定的对象、批次、依赖和范围。',
               'The backend maps communication to Runtime completion; synchronization covers its defined scope.'),
            tr('全局交接需要应用协议；如果还要确认消费结束，必须明确等待消费后的 ACK / credit。',
               'Global handoff needs a protocol; consumption acknowledgment requires an explicit ACK / credit.')])
    return {'jetty-send-recv': f.render(), 'runtime-ubdma-modes': g.render(),
            'jetty-completion-scope': h.render()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    count = 0
    for lang in ('en', 'zh'):
        for name, svg in diagrams(lang).items():
            count += 1
            path = OUT / lang / f'{name}.svg'
            expected = svg.encode('utf-8')
            if args.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(expected)
            elif not path.is_file() or path.read_bytes() != expected:
                stale.append(str(path))
    if stale:
        raise SystemExit('Missing/outdated diagrams; run --write:\n' + '\n'.join(stale))
    print(f'PASS: {count} reproducible bilingual transfer diagrams.')


if __name__ == '__main__':
    main()
