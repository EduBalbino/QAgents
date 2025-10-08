from scripts.core.builders import Recipe, csv, select, device, encoder, ansatz, train, run, save
import datetime as _dt

if __name__ == "__main__":
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    recipe = (
        Recipe()
        | csv('../data/ML-EdgeIIoT-dataset-binario.csv', sample_size=500)
        | select([
        'ip.src_host',
        'ip.dst_host',
        'arp.dst.proto_ipv4',
        'arp.opcode',
        'arp.hw.size',
        'arp.src.proto_ipv4',
        'icmp.checksum',
        'icmp.seq_le',
        'icmp.transmit_timestamp',
        'icmp.unused',
        'http.file_data',
        'http.content_length',
        'http.request.uri.query',
        'http.request.method',
        'http.referer',
        'http.request.full_uri',
        'http.request.version',
        'http.response',
        'http.tls_port',
        'tcp.ack',
        'tcp.ack_raw',
        'tcp.checksum',
        'tcp.connection.fin',
        'tcp.connection.rst',
        'tcp.connection.syn',
        'tcp.connection.synack',
        'tcp.dstport',
        'tcp.flags',
        'tcp.flags.ack',
        'tcp.len',
        'tcp.options',
        'tcp.payload',
        'tcp.seq',
        'tcp.srcport',
        'udp.port',
        'udp.stream',
        'udp.time_delta',
        'dns.qry.name',
        'dns.qry.name.len',
        'dns.qry.qu',
        'dns.qry.type',
        'dns.retransmission',
        'dns.retransmit_request',
        'dns.retransmit_request_in',
        'mqtt.conack.flags',
        'mqtt.conflag.cleansess',
        'mqtt.conflags',
        'mqtt.hdrflags',
        'mqtt.len',
        'mqtt.msg_decoded_as',
        'mqtt.msg',
        'mqtt.msgtype',
        'mqtt.proto_len',
        'mqtt.protoname',
        'mqtt.topic',
        'mqtt.topic_len',
        'mqtt.ver',
        'mbtcp.len',
        'mbtcp.trans_id',
        'mbtcp.unit_id'
        ], label='Attack_label')
        | device("lightning.qubit", wires_from_features=True)
        | encoder("angle_embedding_y")
        | ansatz("ring_rot_cnot", layers=3)
        | train(lr=0.1, batch=64, epochs=1, class_weights="balanced", seed=42)
        | save(f"models/QML_ML-EdgeIIoT-Binario_{ts}.pt")
    )
    run(recipe)
