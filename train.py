# Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio
from sklearn.preprocessing import LabelEncoder

# Load data
dfStudents = pd.read_csv("Data/data.csv", sep=';')
# Suffle to avoid bias
dfStudents = dfStudents.sample(frac=1)

# Remove rows with nulls if it's less than 5% of the data
if ((dfStudents.isna().any(axis=1).sum())/dfStudents.shape[0]) < .05:
    dfStudents.dropna(inplace=True)

# Encode Target
dfStudents['encodedTarget'] = LabelEncoder().fit_transform(dfStudents['Target'])
# Add 1 to avoid 0 bias
dfStudents['encodedTarget']  = dfStudents['encodedTarget'] + 1 

# Train test split
XFeatures = dfStudents.drop(["Target", 'encodedTarget'], axis=1)
yTarget = dfStudents[['encodedTarget']]

# Note: stratify target to avoid unbalanced class bias
XTrain, XTest, yTrain, yTest = train_test_split(
    XFeatures, yTarget, test_size=0.3, random_state=42, stratify=yTarget
)

# MI score
def make_mi_scores(X_MI, y_MI):
    mi_scores = mutual_info_regression(X_MI, y_MI, discrete_features="auto")
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_MI.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mIScores = make_mi_scores(XTrain, yTrain)

# Remove lower mi scores features
selectedFeatures = mIScores[mIScores > 0.01].index.tolist()
XTrain = XTrain[selectedFeatures].copy()
XTest = XTest[selectedFeatures].copy()

# Pipeline
categoricFeatures = XTrain.select_dtypes('object').columns.tolist()
numericFeatures = XTrain.select_dtypes('number').columns.tolist()

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), categoricFeatures),
        ("num_imputer", SimpleImputer(strategy="median"), numericFeatures),
        ("num_scaler", StandardScaler(), numericFeatures),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)
pipe.fit(XTrain, yTrain)

# Model quality metrics

predictions = pipe.predict(XTest)
accuracy = accuracy_score(yTest, predictions)
f1 = f1_score(yTest, predictions, average="macro")

# Save model quality metrics
with open("Results/metrics.txt", "w") as outputFile:
    outputFile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Confusion matrix

confusionMatrix = confusion_matrix(yTest, predictions, labels=pipe.classes_)
confusionFig = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=pipe.classes_)
confusionFig.plot()
plt.savefig("Results/modelResults.png", dpi=120)

# Save model
sio.dump(pipe, "Model/studentsPipeline.skops")