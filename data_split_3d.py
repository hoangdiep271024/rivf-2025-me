# import pandas as pd

# # Lặp qua 5 fold
# for fold in range(1, 6):
#     # Đọc file CSV của fold hiện tại
#     df = pd.read_csv(f"artifacts/casme_split/fold_{fold}/train.csv")
    
#     new_rows = []

#     for _, row in df.iterrows():
#         seq = row["Sequence"]
#         label = row["label"]
#         # Nhân 20 sequence
#         for i in range(1, 21):
#             new_seq = f"{seq}_{i:02d}"
#             new_rows.append([new_seq, label])

#     # Tạo DataFrame mới
#     new_df = pd.DataFrame(new_rows, columns=["Sequence", "label"])
#     # Lưu file mới cho fold tương ứng
#     new_df.to_csv(f"artifacts/casme_split/fold_{fold}/train_new.csv", index=False)

# print("Đã tạo xong tất cả 5 fold!")



import pandas as pd

# Lặp qua 5 fold
for fold in range(1, 6):
    # Đọc file CSV của fold hiện tại
    df = pd.read_csv(f"artifacts/casme_split/fold_{fold}/train.csv")
    
    new_rows = []

    for _, row in df.iterrows():
        seq = row["Sequence"]
        label = row["label"]
        # Giữ nguyên sequence gốc
        new_rows.append([seq, label])
        # Thêm 20 sequence mới
        for i in range(1, 21):
            new_seq = f"{seq}_{i:02d}"
            new_rows.append([new_seq, label])

    # Tạo DataFrame mới
    new_df = pd.DataFrame(new_rows, columns=["Sequence", "label"])
    # Lưu file mới cho fold tương ứng
    new_df.to_csv(f"artifacts/casme_split/fold_{fold}/train_new.csv", index=False)

print("Đã tạo xong tất cả 5 fold!")

