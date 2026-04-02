from src.preprocess import preprocess_pipeline, save_processed_data
from src.train import train_and_select_best_model


def run_pipeline():
    print("Step 1: Preprocessing...")
    train_df, val_df, test_df = preprocess_pipeline("data/raw/train_FD001.txt")
    save_processed_data(train_df, val_df, test_df)

    print("\nStep 2: Training...")
    best_name, _, results = train_and_select_best_model()

    best_val = results[best_name]["validation"]
    best_test = results[best_name]["test"]

    print("\n==============================")
    print("🏆 FINAL BEST MODEL:", best_name)
    print("==============================")

    print("\n📊 VALIDATION PERFORMANCE:")
    print(f"Accuracy: {best_val['accuracy']:.4f}")
    print(f"F1 Score: {best_val['f1']:.4f}")
    print(f"Confusion Matrix: {best_val['confusion_matrix']}")

    print("\n📊 TEST PERFORMANCE:")
    print(f"Accuracy: {best_test['accuracy']:.4f}")
    print(f"F1 Score: {best_test['f1']:.4f}")
    print(f"Confusion Matrix: {best_test['confusion_matrix']}")


if __name__ == "__main__":
    run_pipeline()