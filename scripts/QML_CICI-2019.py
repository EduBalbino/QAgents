from scripts.core.builders import Recipe, csv, select, device, encoder, ansatz, train, run

if __name__ == "__main__":
    recipe = (
        Recipe()
        | csv('data/PCA_CIC-DDoS2019.csv', sample_size=500)
        | select([f'PC_{i+1}' for i in range(8)], label='Label')
        | device("lightning.qubit", wires_from_features=True)
        | encoder("angle_embedding_y")
        | ansatz("ring_rot_cnot", layers=3)
        | train(lr=0.1, batch=64, epochs=1, class_weights="balanced", seed=42)
    )
    run(recipe)