import pandas as pd

def count_classes(csv_path, label_col="label"):
    """
    Đếm số lượng mẫu của từng class trong file CSV.
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{label_col}' trong CSV")
    
    counts = df[label_col].value_counts()  # Số lượng mỗi class
    num_classes = df[label_col].nunique()  # Số class khác nhau
    
    print(f"File: {csv_path}")
    print(f"Số class: {num_classes}")
    print("Số lượng mẫu mỗi class:")
    print(counts)
    return counts, num_classes

def main():
    # Tính số class cho train và valid
    train_counts, train_num = count_classes("data_csv/label_casme_goc_full.csv")
    valid_counts, valid_num = count_classes("data_csv/label_sam_goc_full.csv")
    
    print("\n--- Tổng quan ---")
    print(f"Train: {train_num} classes")
    print(f"Valid: {valid_num} classes")

if __name__ == "__main__":
    main()

