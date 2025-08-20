from scripts.core.builders import Recipe, csv, select, device, encoder, ansatz, train, run

if __name__ == "__main__":
    recipe = (
        Recipe()
        | csv('data/ML-EdgeIIoT-dataset.csv', sample_size=500)
        | select([
            'icmp.checksum',
            'icmp.seq_le',
            'http.content_length',
            'http.response',
            'tcp.ack',
            'tcp.len',
            'tcp.seq',
            'tcp.dstport'
        ], label='Attack_label')
        | device("lightning.qubit", wires_from_features=True)
        | encoder("angle_embedding_y")
        | ansatz("ring_rot_cnot", layers=3)
        | train(lr=0.1, batch=64, epochs=1, class_weights="balanced", seed=42)
    )
    run(recipe)
