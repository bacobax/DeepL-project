from training_system import CoCoOpSystem

if __name__ == "__main__":
    train_sys = CoCoOpSystem(
        dataset_name="cifar10",
        batch_size=16,
        num_classes=10,
        device="mps",
        learning_rate=0.002,
        weight_decay=0.0005,
        momentum=0.9,
        epochs=2,
        run_name="exp1",
        n_ctx=4,
        ctx_init="",
        class_token_position="end",
        csc=False,
    )
    
    train_sys.train()
