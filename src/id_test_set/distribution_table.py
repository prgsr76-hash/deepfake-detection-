import pandas as pd

id_test = pd.read_csv('id_test.csv')

print("=== ID Test Set Distribution ===\n")

print(f"Total frames: {len(id_test)}")
print(f"Total unique videos: {id_test['video_id'].nunique()}\n")

print("By label (0=real, 1=fake):")
print(id_test['label'].value_counts())
print()

print("By manipulation type:")
print(id_test['manipulation'].value_counts())
print()

print("Videos per manipulation type:")
print(id_test.groupby('manipulation')['video_id'].nunique())
print()

print("Frames per manipulation type:")
print(id_test.groupby('manipulation').size())
import pandas as pd

id_test = pd.read_csv('id_test.csv')

summary = pd.DataFrame({
    'Category': ['Real', 'Fake - DeepFakes', 'Fake - FaceSwap'],
    'Videos': [
        id_test[id_test['manipulation']=='real']['video_id'].nunique(),
        id_test[id_test['manipulation']=='DeepFakes']['video_id'].nunique(),
        id_test[id_test['manipulation']=='FaceSwap']['video_id'].nunique(),
    ],
    'Frames': [
        len(id_test[id_test['manipulation']=='real']),
        len(id_test[id_test['manipulation']=='DeepFakes']),
        len(id_test[id_test['manipulation']=='FaceSwap']),
    ]
})

summary.to_csv('id_test_distribution_table.csv', index=False)
print(summary)
