# Author: Abeer Mansour

import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


files = [
    "test_novelty_generation/test_outputs_systematic_500/openai__gpt-oss-20b/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/maxidl__Llama-OpenReviewer-8B/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/meta-llama__Llama-3.1-8B-Instruct/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/mistralai__Mistral-7B-Instruct-v0.1/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/Qwen__Qwen2.5-14B-Instruct-1M/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/SenthilKumarN__SciLlama-3.2-3B/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/weathon__paper_reviewer/model_results.parquet",
    "test_novelty_generation/test_outputs_systematic_500/AbeerMostafa__Novelty_Reviewer/model_results.parquet"
]


scores = []
for file in files:
    df = pl.read_parquet(file)
    
    df_matched = df.with_columns([
        pl.col("generated_text")
        .str.replace_all("*", "", literal=True)
        .str.extract(r"(?i)Novelty Score:\s*(-?\d+)", 1)
        .cast(pl.Int8)
        .alias("result_score")
    ])

    df_clean = df_matched.filter(
        pl.col("result_score").is_not_nan() &
        pl.col("novelty_score").is_not_nan()
    )
    num_matches = df_clean.filter(pl.col('result_score') == pl.col('novelty_score')).height
    total_rows = df_clean.height

    y_pred = df_clean["result_score"].to_numpy()
    y_true = df_clean["novelty_score"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average="macro")
    recall = recall_score(y_true, y_pred, zero_division=0, average="macro")
    pr, _ = pearsonr(df_clean['result_score'], df_clean['novelty_score'])
    sr, _ = spearmanr(df_clean['result_score'], df_clean['novelty_score'])
    kt, _ = kendalltau(df_clean['result_score'], df_clean['novelty_score'])
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{'='*60}")
    print(f"Processing Model: {file.split('/')[-2]}")
    print(f"{'='*60}")
    print(f"Exact matches: {num_matches} out of 500")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Pearson correlation: {pr:.4f}")
    print(f"Spearman correlation: {sr:.4f}")
    print(f"Kendall Tau correlation: {kt:.4f}\n")
    print(f"F1 Score: {f1:.4f}\n")
    #print(df_matched['result_score'].to_list())
    #print(df_matched['novelty_score'].to_list())

    #example_row = df.select("generated_text").head(1)
    #print("Example generated text:", example_row["generated_text"][0])

    result = (df.with_columns([
        pl.col("generated_text")
        .str.replace_all("*", "", literal=True)
        .str.extract(r"(?i)Novelty Score:\s*(-?\d+)", 1)
        .cast(pl.Int8)
        .alias("result_score")
    ])
    .filter(pl.col("result_score").is_not_null())  
    ["result_score"].to_list())
    scores.append(result)



scores.append(pl.read_parquet(files[0])["novelty_score"].drop_nulls().to_list())

labels = [
    'GPT-OSS-20B',
    'OpenReviewer',
    'Llama 3.1 8B',
    'Mistral 7B',
    'Qwen 2.5 14B',
    'SciLlama',
    'Paper Reviewer',
    'Novelty Reviewer',
    'Ground Truth'
]

colors = [
    '#4C72B0',  # muted blue
    '#55A868',  # muted green
    '#C44E52',  # muted red
    '#8172B2',  # muted purple
    '#64B5CD',  # teal
    '#CCB974',  # muted yellow
    '#8C8C8C',  # neutral gray
    '#DD8452',  # soft orange
    '#937860'   # brown
]

x = np.array([-1, 0, 1, 2])
bar_height = 0.14  # thickness of horizontal bars

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(13, 8))

# --- light horizontal background bands for each group (-1,0,1,2) ---
band_colors = ["#dbe7ff", "#ddf6e5", "#ffe6d6", "#dfdae6"]  # more visible pastels
group_h = len(scores) * bar_height
pad = bar_height * 0.8

for pos, bc in zip(x, band_colors):
    y0 = pos * 1.6 - pad
    y1 = pos * 1.6 + group_h + pad
    ax.axhspan(y0, y1, color=bc, alpha=0.65, zorder=-5)

# Plot horizontal bars
for i, (score_list, color, label) in enumerate(zip(scores, colors, labels)):
    score_list = score_list + [1] * (500 - len(score_list))
    counts = [score_list.count(val) for val in x]

    ax.barh(
        y=[pos * 1.6 + i * bar_height for pos in x],
        width=counts,
        height=bar_height,
        color=color,
        alpha=0.9,
        label=label,
        edgecolor="white",
        linewidth=0.6,
        zorder=2
    )

# Axis labels and ticks (clear & paper-ready)
ax.set_ylabel("Novelty Score", fontsize=16, fontweight="bold")
ax.set_xlabel("Count", fontsize=16, fontweight="bold")

ax.set_yticks([pos * 1.6 + bar_height * 4 for pos in x])
ax.set_yticklabels(x, fontsize=14)
ax.tick_params(axis="x", labelsize=13)

# Grid (horizontal bars → grid on x-axis)
ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.3)
ax.set_axisbelow(True)

# Legend: bigger, clearer, paper-style
ax.legend(
    loc="lower right",
    ncol=3,
    fontsize=16,
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    edgecolor="0.8"
)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

plt.tight_layout()
plt.savefig(
    "novelty_scores_histogram_horizontal.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.show()



labels = [
    'GPT-OSS-20B',
    'OpenReviewer',
    'Llama 3.1 8B',
    'Mistral 7B',
    'Qwen 2.5 14B',
    'SciLlama',
    'Paper Reviewer',
    'Novelty Reviewer',
    'Ground Truth'
]


# Process data: pad to 500 and calculate percentages
negative_pcts = []  # -1
z_pcts = []      # 0
moderate_pcts = []  #1
positive_pcts = []  #2

for score_list in scores:
    # Pad to 500 samples
    total = len(score_list)
    
    # Count occurrences
    count_neg1 = score_list.count(-1)
    count_0 = score_list.count(0)
    count_1 = score_list.count(1)
    count_2 = score_list.count(2)
    
    # Calculate percentages
    negative_pct = ((count_neg1) / total) * 100
    z_pct = (count_0 / total) * 100
    moderate_pct = (count_1 / total) * 100
    positive_pct = (count_2 / total) * 100

    negative_pcts.append(negative_pct)
    z_pcts.append(z_pct)
    moderate_pcts.append(moderate_pct)
    positive_pcts.append(positive_pct)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(labels))

# Colors for negative and positive
color_negative = '#750C10'  # dark red for negative (-1)
color_zero = '#E57373'  # soft red for zero (0)
color_moderate = '#FFB74D'  # soft orange for moderate (1)
color_positive = '#81C784'  # dark green for positive (2)


# Create horizontal stacked bars
ax.barh(y_pos, negative_pcts, color=color_negative, edgecolor='white', label='Not novel (-1)', zorder=3)
ax.barh(y_pos, z_pcts, left=negative_pcts, color=color_zero, edgecolor='white', label='Limited Novelty (0)', zorder=3)
ax.barh(y_pos, moderate_pcts, left=np.array(negative_pcts) + np.array(z_pcts), color=color_moderate, edgecolor='white', label='Moderate Novelty (1)', zorder=3)
ax.barh(y_pos, positive_pcts, left=np.array(negative_pcts) + np.array(z_pcts) + np.array(moderate_pcts), color=color_positive, edgecolor='white', label='High Novelty (2)', zorder=3)

# Customize axes
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=18, fontweight='bold')
ax.set_xlabel('Percentage (%)', fontsize=18, fontweight='bold')
ax.set_xlim(0, 100)
# Grid and styling
ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Legend
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),  # move legend below plot
    ncol=2,                      # number of labels in one row
    frameon=False,                # or False if you want no box
     prop={'weight': 'bold', 'size': 14},
     handlelength=1.5
)

plt.tight_layout()
plt.savefig('model_assessments_stacked.pdf', dpi=300, bbox_inches='tight')
plt.show()


labels = [
    'GPT-OSS-20B',
    'OpenReviewer',
    'Llama 3.1 8B',
    'Mistral 7B',
    'Qwen 2.5 14B',
    'SciLlama',
    'Paper Reviewer',
    'Novelty Reviewer',
    'Ground Truth'
]

colors = [
    '#4C72B0',  # muted blue
    '#55A868',  # muted green
    '#C44E52',  # muted red
    '#8172B2',  # muted purple
    '#64B5CD',  # teal
    '#CCB974',  # muted yellow
    '#8C8C8C',  # neutral gray
    '#DD8452',  # soft orange
    '#937860'   # brown
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for idx, (score_list, color, label) in enumerate(zip(scores, colors, labels)):
    ax = axes[idx]
    score_list = score_list + [1] * (500 - len(score_list))
    # Create histogram with KDE overlay
    counts, bins, patches = ax.hist(
        score_list, 
        bins=[-1.5, -0.5, 0.5, 1.5, 2.5],
        alpha=0.7, 
        color=color,
        edgecolor='white',
        linewidth=1.5
    )
    
    # Calculate statistics
    mean_val = np.mean(score_list)
    median_val = np.median(score_list)
    mode_val = stats.mode(score_list, keepdims=True)[0][0]
    skew_val = stats.skew(score_list)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}', alpha=0.8)
    ax.axvline(median_val, color='darkblue', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}', alpha=0.8)
    
    # Title with model name and skew
    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Novelty Score', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_xticks([-1, 0, 1, 2])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_facecolor('#fafafa')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('model_distributions_skew.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()



'''
for i in range(20):
    row = df_result.row(i, named=True) 
    for column, value in row.items():
        if column in ['generated_text', 'result_score', 'novelty_score']:
            print(f"{column}: {value}")
    print("\n" + "="*50 + "\n")

null_vals = df_result.filter(pl.col('result_score').is_null())

for i in range(len(null_vals)):
    row = null_vals.row(i, named=True) 
    for column, value in row.items():
        if column in ['generated_text']:
            print(f"{column}: {value}")
    print("\n" + "="*50 + "\n")
'''


