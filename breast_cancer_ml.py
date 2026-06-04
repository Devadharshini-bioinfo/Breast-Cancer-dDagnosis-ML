# ============================================================
#  BREAST CANCER DIAGNOSIS USING MACHINE LEARNING
#  Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
#  Author: [Your Name]
#  Task: Binary Classification — Malignant vs Benign
# ============================================================

# ── STEP 1: INSTALL / IMPORT LIBRARIES ──────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    f1_score, recall_score, precision_score
)
from sklearn.decomposition import PCA

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

print("✅ All libraries loaded successfully!\n")


# ── STEP 2: LOAD DATASET ────────────────────────────────────
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target          # 0 = Malignant, 1 = Benign
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"  Total samples  : {df.shape[0]}")
print(f"  Total features : {df.shape[1] - 2}")
print(f"  Benign (1)     : {(df.target == 1).sum()} samples")
print(f"  Malignant (0)  : {(df.target == 0).sum()} samples")
print(f"  Missing values : {df.isnull().sum().sum()}")
print("=" * 55)
print("\nFirst 5 rows (selected features):")
print(df[['mean radius', 'mean texture', 'mean perimeter',
          'mean area', 'mean smoothness', 'diagnosis']].head())


# ── STEP 3: PREPROCESSING ───────────────────────────────────
X = df[data.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✅ Train set : {X_train_scaled.shape[0]} samples")
print(f"✅ Test set  : {X_test_scaled.shape[0]} samples")


# ── STEP 4: TRAIN 4 MODELS ──────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                : SVC(kernel='rbf', probability=True, random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=5)
}

results = {}

print("\n" + "=" * 55)
print("MODEL TRAINING & EVALUATION")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc   = auc(fpr, tpr)
    cv_score  = cross_val_score(model, X_train_scaled, y_train,
                                cv=5, scoring='accuracy').mean()

    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
        'accuracy' : acc,
        'f1'       : f1,
        'recall'   : recall,
        'precision': precision,
        'auc'      : roc_auc,
        'cv_score' : cv_score,
        'fpr'      : fpr,
        'tpr'      : tpr,
        'cm'       : confusion_matrix(y_test, y_pred)
    }

    print(f"\n  {name}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    AUC-ROC   : {roc_auc:.4f}")
    print(f"    CV Score  : {cv_score:.4f}")


# ── STEP 5: SUMMARY TABLE ───────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY TABLE")
print("=" * 55)
summary = pd.DataFrame({
    name: {
        'Accuracy' : r['accuracy'],
        'F1'       : r['f1'],
        'Recall'   : r['recall'],
        'Precision': r['precision'],
        'AUC-ROC'  : r['auc'],
        'CV Score' : r['cv_score']
    }
    for name, r in results.items()
}).T.round(4)
print(summary.to_string())


# ── STEP 6: VISUALIZATIONS ──────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

# 6A: Class Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['diagnosis'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=['#E53935', '#43A047'], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Diagnosis')
plt.tight_layout()
plt.savefig('plot_01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_01_class_distribution.png")

# 6B: Feature Correlation Heatmap
top_features = df[data.feature_names[:15]]
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(top_features.corr(), dtype=bool))
sns.heatmap(top_features.corr(), mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={'size': 7}, ax=ax)
ax.set_title('Feature Correlation Heatmap (Top 15 Features)',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plot_02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_02_correlation_heatmap.png")

# 6C: PCA 2D Projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(scaler.fit_transform(X))
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['diagnosis'] = df['diagnosis'].values

fig, ax = plt.subplots(figsize=(8, 6))
for label, color in zip(['Benign', 'Malignant'], ['#43A047', '#E53935']):
    mask = pca_df['diagnosis'] == label
    ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
               c=color, label=label, alpha=0.65, edgecolors='white',
               linewidths=0.4, s=55)
ax.set_title('PCA — 2D Feature Space Projection', fontsize=13,
             fontweight='bold', pad=12)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig('plot_03_pca.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_03_pca.png")

# 6D: Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, r) in zip(axes.flatten(), results.items()):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                linewidths=1, ax=ax, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{name}\nAcc: {r["accuracy"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
fig.suptitle('Confusion Matrices — All Models', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_04_confusion_matrices.png")

# 6E: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
for (name, r), color in zip(results.items(), COLORS):
    ax.plot(r['fpr'], r['tpr'], color=color, lw=2,
            label=f"{name} (AUC = {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random classifier')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='lower right', framealpha=0.9)
plt.tight_layout()
plt.savefig('plot_05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_05_roc_curves.png")

# 6F: Model Comparison
metrics = ['accuracy', 'f1', 'recall', 'precision', 'auc']
metric_labels = ['Accuracy', 'F1', 'Recall', 'Precision', 'AUC-ROC']
x = np.arange(len(metrics))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=name,
           color=COLORS[i], edgecolor='white', linewidth=0.8)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim([0.88, 1.01])
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=13,
             fontweight='bold', pad=12)
ax.legend(framealpha=0.9)
ax.axhline(y=0.95, color='gray', linestyle='--', lw=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig('plot_06_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_06_model_comparison.png")

# 6G: Feature Importance
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_,
                        index=data.feature_names).sort_values(ascending=True)
top10 = importances.tail(10)

fig, ax = plt.subplots(figsize=(8, 6))
colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))
top10.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='white')
ax.set_title('Top 10 Most Important Features\n(Random Forest)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Feature Importance Score')
plt.tight_layout()
plt.savefig('plot_07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_07_feature_importance.png")

# 6H: SHAP Summary Plot
if SHAP_AVAILABLE:
    print("\nGenerating SHAP values (may take ~30s)...")
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_test_scaled,
                      feature_names=data.feature_names,
                      plot_type='dot', show=False, max_display=15)
    plt.title('SHAP Summary Plot — Random Forest',
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('plot_08_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: plot_08_shap_summary.png")
else:
    print("⚠️  Skipping SHAP. Install with: pip install shap")


# ── STEP 7: BEST MODEL REPORT ───────────────────────────────
best_model_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_model_name]

print("\n" + "=" * 55)
print(f"🏆 BEST MODEL: {best_model_name}")
print("=" * 55)
print(classification_report(y_test, best['y_pred'],
                             target_names=['Malignant', 'Benign']))
print("\n✅ ALL DONE! Check your directory for 7–8 PNG plots.")
